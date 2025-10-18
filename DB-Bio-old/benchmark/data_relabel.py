import json
import asyncio
from pathlib import Path
from typing import List, Dict
import aiohttp
from tqdm.asyncio import tqdm_asyncio
import argparse

class DataRelabeler:
    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com/v1"):
        """
        初始化数据重标注器
        
        Args:
            api_key: DeepSeek API密钥
            api_base: API基础URL
        """
        self.api_key = api_key
        self.api_base = api_base
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_accurate_label(self, session: aiohttp.ClientSession, text: str, current_label: str) -> str:
        """
        调用DeepSeek API获取精确标签
        
        Args:
            session: aiohttp会话
            text: 文本内容
            current_label: 当前标签
            
        Returns:
            精确的标签
        """
        prompt = f"""Based on the following biographical text, evaluate whether a more accurate and specific professional label can be determined.

        Text Content:
        {text}

        Current Label: {current_label}

        Instructions:
        1. Carefully read the text and identify the person's primary occupation or professional identity
        2. Compare it with the current label
        3. If you can determine a MORE PRECISE and SPECIFIC label that better represents their profession, provide the new label
        4. If the current label is already accurate and appropriate, return the EXACT SAME label
        5. The label should be a single word or short phrase in English
        6. Do not add any explanation or additional text

        Your Label:"""

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 50
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        label = result["choices"][0]["message"]["content"].strip()
                        return label
                    else:
                        error_text = await response.text()
                        print(f"API错误 (状态码 {response.status}): {error_text}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            return None  # 失败后返回None标记
            except asyncio.TimeoutError:
                print(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
            except Exception as e:
                print(f"请求异常: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    async def process_item(self, session: aiohttp.ClientSession, item: Dict) -> Dict:
        """
        处理单个数据项
        
        Args:
            session: aiohttp会话
            item: 数据项
            
        Returns:
            添加了精确标签的数据项
        """
        text = item.get("text", "")
        current_label = item.get("label", "")
        
        # 如果已经有label_accurate且不为None,跳过处理
        if "label_accurate" in item and item["label_accurate"] is not None:
            return item
        
        if not text or not current_label:
            item["label_accurate"] = current_label
            return item
        
        accurate_label = await self.get_accurate_label(session, text, current_label)
        item["label_accurate"] = accurate_label  # None表示处理失败
        return item
    
    async def process_batch(self, items: List[Dict], batch_size: int = 10) -> List[Dict]:
        """
        批量处理数据项
        
        Args:
            items: 数据项列表
            batch_size: 批处理大小
            
        Returns:
            处理后的数据项列表
        """
        connector = aiohttp.TCPConnector(limit=batch_size)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.process_item(session, item) for item in items]
            results = await tqdm_asyncio.gather(*tasks, desc="处理数据")
            return results
    
    def process_file(self, input_path: str, output_path: str, batch_size: int = 10):
        """
        处理JSONL文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            batch_size: 批处理大小(并行请求数)
        """
        # 读取数据
        print(f"正在读取文件: {input_path}")
        items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        print(f"共读取 {len(items)} 条数据")
        
        # 检查输出文件是否存在,如果存在则加载已处理的数据
        if Path(output_path).exists():
            print(f"检测到已存在的输出文件,加载已处理数据...")
            processed_items = []
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        processed_items.append(json.loads(line))
            
            # 将已处理的label_accurate合并到items中
            if len(processed_items) == len(items):
                for i, processed in enumerate(processed_items):
                    if "label_accurate" in processed:
                        items[i]["label_accurate"] = processed["label_accurate"]
                print(f"已加载 {len(processed_items)} 条已处理数据")
        
        # 统计需要处理的数量
        need_process = sum(1 for item in items 
                          if "label_accurate" not in item or item["label_accurate"] is None)
        print(f"需要处理 {need_process} 条数据 (跳过 {len(items) - need_process} 条)")
        
        # 异步处理
        print(f"开始处理数据 (并行数: {batch_size})...")
        results = asyncio.run(self.process_batch(items, batch_size))
        
        # 写入结果
        print(f"正在写入结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print("处理完成!")
        
        # 统计信息
        changed_count = sum(1 for item in results 
                          if item.get("label") != item.get("label_accurate") 
                          and item.get("label_accurate") is not None)
        failed_count = sum(1 for item in results 
                          if item.get("label_accurate") is None)
        success_count = sum(1 for item in results 
                           if item.get("label_accurate") is not None)
        
        print(f"标签变化数量: {changed_count}/{len(results)}")
        print(f"处理成功: {success_count}, 处理失败: {failed_count}")
        
        if failed_count > 0:
            print(f"\n提示: 有 {failed_count} 条数据处理失败(label_accurate为None)")
            print("请重新运行脚本以重试失败的数据")


def main():
    parser = argparse.ArgumentParser(description="使用DeepSeek API对数据集进行精确标签标注")
    parser.add_argument("--input", "-i", required=True, help="输入JSONL文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出JSONL文件路径")
    parser.add_argument("--api-key", "-k", required=True, help="DeepSeek API密钥")
    parser.add_argument("--batch-size", "-b", type=int, default=10, 
                       help="并行处理的批大小 (默认: 10)")
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1",
                       help="API基础URL (默认: https://api.deepseek.com/v1)")
    
    args = parser.parse_args()
    
    # 创建重标注器
    relabeler = DataRelabeler(api_key=args.api_key, api_base=args.api_base)
    
    # 处理文件
    relabeler.process_file(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()