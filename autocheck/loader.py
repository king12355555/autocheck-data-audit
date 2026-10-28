# -*- coding: utf-8 -*-
"""数据读取: 支持 CSV / TSV / TXT / Excel, 自动处理编码与分隔符。"""
import io
import os

import pandas as pd

_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030"]


def _guess_sep(text):
    first = text.splitlines()[0] if text else ""
    best, best_n = None, -1
    for sep in [",", "\t", ";", "|"]:
        n = first.count(sep)
        if n > best_n:
            best, best_n = sep, n
    return best or ","


def _decode(raw):
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def read_table(source):
    """source 可以是路径字符串或 file-like(Streamlit 上传对象)。返回 DataFrame。"""
    name = ""
    if isinstance(source, (str, os.PathLike)):
        name = str(source)
        if not os.path.exists(source):
            raise FileNotFoundError(f"文件不存在: {source}")
        ext = os.path.splitext(name)[1].lower()
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(source)
        with open(source, "rb") as f:
            raw = f.read()
    else:
        name = getattr(source, "name", "") or ""
        ext = os.path.splitext(name)[1].lower()
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(source)
        raw = source.read()

    text = _decode(raw)
    df = pd.read_csv(io.StringIO(text), sep=_guess_sep(text), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df
