"""文本切块层。先按 Markdown 标题组织，再按字符预算切分，返回携带元数据的 Document 列表。"""

import collections

from dotenv import load_dotenv
import  os
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document
if __package__:
    from .embeddings import TongyiEmbeddings
    from .zipload import zip_markdown
    from .vector_store import save_chunks
else:
    from embeddings import TongyiEmbeddings
    from zipload import zip_markdown
    from vector_store import save_chunks

def prepare_chunks(
    markdown: str,
    paper_id: str,
    source: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> list[Document]:
    """
    将 Markdown 文本按标题和长度切块，
    并为每个 Chunk 添加元数据。
    """

    # 1. 检查输入
    if not isinstance(markdown, str):
        raise TypeError("markdown 必须是字符串")

    if not markdown.strip():
        return []

    # 2. 定义 Markdown 标题层级
    headers = [
        ("#", "chapter"),
        ("##", "section"),
        ("###", "subsection")
    ]  # 标题层级写入对应元数据字段，便于后续展示章节归属。

    # 3. 第一阶段：按标题切分
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False  # Chunk 正文保留标题文字，使向量化时携带章节语义。
    )

    docs = md_splitter.split_text(markdown)

    print(f"第一次切分：{len(docs)} 个章节")

    # 4. 第二阶段：按字符长度切分
    text_splitter = RecursiveCharacterTextSplitter(
        # 优先按段落、换行和中文句末拆分，最后使用字符级边界兜底。
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    chunks = text_splitter.split_documents(docs)

    print(f"第二次切分：{len(chunks)} 个 Chunk")

    # 5. 为每个 Chunk 添加元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "paper_id": paper_id,
            "source": source,
            "chunk_id": f"{paper_id}_{i}",
            "chunk_index": i
        })

    return chunks


def main():
    """运行历史 ZIP Markdown 示例，打印 Chunk 与元数据，再执行向量化和保存。"""
    load_dotenv()
    # 1. 调用你已经写好的 ZIP 解析函数
    markdown = zip_markdown()

    # 如果你的函数需要 ZIP 路径，
    # 就改成 zip_markdown(zip_path)

    # 2. 处理 Markdown
    chunks = prepare_chunks(
        markdown=markdown,
        paper_id="paper001",
        source="langchaimquestion.pdf"
    )

    # 3. 查看处理结果
    print("\n========== 切分结果 ==========")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n===== Chunk {i} =====")
        print("文本内容：")
        print(chunk.page_content[:500])
        print("\n元数据：")
        print(chunk.metadata)

    # 4. 提取将来需要 Embedding 的文本
    texts = [
        chunk.page_content
        for chunk in chunks
    ]
    print(f"\n待向量化文本数量：{len(texts)},{len(chunks)}")
    embeddings = TongyiEmbeddings(api_key=os.getenv("DASHSCOPE_API_KEY")).embed_documents(texts)
    print({len(embeddings)})
    collections = save_chunks(chunks, embeddings)
    return collections



if __name__ == "__main__":
    chunks = main()
