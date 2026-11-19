"""纯文本切块层，按 Markdown 标题与字符长度拆分并附加基础元数据。"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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
