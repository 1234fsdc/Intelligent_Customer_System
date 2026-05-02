import pandas as pd
import os
from langchain_core.tools import tool
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEVICES_DIR = BASE_DIR / "data" / "devices"

devices_data = {
    "phone": None,
    "tablet": None,
    "computer": None
}

def load_device_data():
    global devices_data
    device_files = {
        "phone": DEVICES_DIR / "phones.csv",
        "tablet": DEVICES_DIR / "tablets.csv",
        "computer": DEVICES_DIR / "computers.csv"
    }
    for device_type, file_path in device_files.items():
        if file_path.exists():
            devices_data[device_type] = pd.read_csv(file_path)
        else:
            devices_data[device_type] = pd.DataFrame()

load_device_data()

@tool(description="根据用户需求检索匹配的电子设备。支持手机、平板、电脑。参数：device_type(设备类型: phone/tablet/computer), query(用户需求描述)")
def search_devices(device_type: str, query: str) -> str:
    """根据用户需求检索匹配的电子设备"""
    if device_type not in devices_data or devices_data[device_type].empty:
        return f"未找到{device_type}类型的设备数据"
    
    df = devices_data[device_type]
    results = []
    query_lower = query.lower()
    
    for _, row in df.iterrows():
        score = 0
        row_text = " ".join([str(v) for v in row.values]).lower()
        
        if query_lower in row_text:
            score += 10
        if "游戏" in query and ("游戏" in str(row.get("使用场景", "")) or "游戏" in str(row.get("适用人群", ""))):
            score += 5
        if "办公" in query and ("办公" in str(row.get("使用场景", "")) or "商务" in str(row.get("适用人群", ""))):
            score += 5
        if "学生" in query and "学生" in str(row.get("适用人群", "")):
            score += 5
        if "拍照" in query and ("拍照" in str(row.get("使用场景", "")) or "影像" in str(row.get("特色功能", ""))):
            score += 5
        if "便携" in query and ("便携" in str(row.get("使用场景", "")) or "轻薄" in str(row.get("分类标签", ""))):
            score += 5
        if "性价比" in query and "性价比" in str(row.get("分类标签", "")):
            score += 5
        
        if score > 0:
            results.append((score, row))
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    if not results:
        return "未找到匹配的设备，请尝试调整搜索条件"
    
    output = []
    for score, row in results[:5]:
        output.append(f"设备: {row['设备名称']}")
        output.append(f"价格: {row['价格']}元")
        output.append(f"适用人群: {row['适用人群']}")
        output.append(f"特色功能: {row['特色功能']}")
        output.append(f"使用场景: {row['使用场景']}")
        output.append(f"分类: {row['分类标签']}")
        output.append("---")
    
    return "\n".join(output)

@tool(description="根据预算范围筛选电子设备。参数：device_type(设备类型: phone/tablet/computer), min_price(最低价格), max_price(最高价格)")
def filter_by_budget(device_type: str, min_price: int = 0, max_price: int = 999999) -> str:
    """根据预算范围筛选电子设备"""
    if device_type not in devices_data or devices_data[device_type].empty:
        return f"未找到{device_type}类型的设备数据"
    
    df = devices_data[device_type]
    filtered = df[(df["价格"] >= min_price) & (df["价格"] <= max_price)]
    
    if filtered.empty:
        return f"在{min_price}-{max_price}元价格区间内没有找到{device_type}设备"
    
    output = []
    for _, row in filtered.iterrows():
        output.append(f"设备: {row['设备名称']}")
        output.append(f"价格: {row['价格']}元")
        output.append(f"适用人群: {row['适用人群']}")
        output.append(f"特色功能: {row['特色功能']}")
        output.append("---")
    
    return "\n".join(output)

@tool(description="根据使用场景匹配电子设备。参数：device_type(设备类型: phone/tablet/computer), scene(使用场景: 游戏/办公/学习/便携/拍照/创作/商务/娱乐)")
def match_by_scene(device_type: str, scene: str) -> str:
    """根据使用场景匹配电子设备"""
    if device_type not in devices_data or devices_data[device_type].empty:
        return f"未找到{device_type}类型的设备数据"
    
    df = devices_data[device_type]
    results = []
    
    for _, row in df.iterrows():
        if scene in str(row.get("使用场景", "")) or scene in str(row.get("适用人群", "")):
            results.append(row)
    
    if not results:
        return f"未找到适合{scene}场景的{device_type}设备"
    
    output = []
    for row in results[:5]:
        output.append(f"设备: {row['设备名称']}")
        output.append(f"价格: {row['价格']}元")
        output.append(f"适用人群: {row['适用人群']}")
        output.append(f"特色功能: {row['特色功能']}")
        output.append(f"使用场景: {row['使用场景']}")
        output.append("---")
    
    return "\n".join(output)

@tool(description="按性价比排序推荐电子设备。参数：device_type(设备类型: phone/tablet/computer), sort_by(排序方式: price_asc价格从低到高/price_desc价格从高到低)")
def sort_by_value(device_type: str, sort_by: str = "price_asc") -> str:
    """按性价比排序推荐电子设备"""
    if device_type not in devices_data or devices_data[device_type].empty:
        return f"未找到{device_type}类型的设备数据"
    
    df = devices_data[device_type].copy()
    
    if sort_by == "price_asc":
        df = df.sort_values("价格", ascending=True)
    elif sort_by == "price_desc":
        df = df.sort_values("价格", ascending=False)
    
    output = []
    for _, row in df.head(5).iterrows():
        output.append(f"设备: {row['设备名称']}")
        output.append(f"价格: {row['价格']}元")
        output.append(f"适用人群: {row['适用人群']}")
        output.append(f"特色功能: {row['特色功能']}")
        output.append(f"使用场景: {row['使用场景']}")
        output.append("---")
    
    return "\n".join(output)
