"""历史 ZIP Markdown 读取示例。直接在压缩包内选择主 Markdown，供 paper_chunk.py 的样例入口使用。"""

from pathlib import Path
from zipfile import ZipFile

def zip_markdown():
    """从样例 ZIP 中选择 full.md 或 markdown.md，按 UTF-8 BOM 兼容方式解码正文。"""
    zip_path = Path(r"C:\Users\Mark\Desktop\langgraph\articels\testresuly.zip")

    with ZipFile(zip_path, "r") as z:
        # 查看压缩包里的文件
        names = z.namelist()
        print(names)

        # 找到 Markdown 文件
        md_files = [
            name for name in names
            if name.lower().endswith(".md")
        ]

        preferred = next(
            # 按既有样例约定选择完整 Markdown，ZIP 内其余产物继续保留供检查。
            (
                name for name in md_files
                if Path(name).name in ("full.md", "markdown.md")
            ),
            None
        )

        if preferred is None:
            raise ValueError("没有找到主 Markdown 文件")

        markdown = z.read(preferred).decode("utf-8-sig")
        # utf-8-sig 处理文件开头的 BOM，标题切块器可以直接读取正文。
        return markdown

if __name__ == '__main__':
    zip_markdown()
