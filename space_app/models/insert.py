from typing import Optional, List, Dict, Any
import re
class ASTNode:
    def __init__(self, node_type):
        self.node_type = node_type
    def to_dict(self):
        pass
class Insert_Node(ASTNode):
    def __init__(self):
        super().__init__('INSERT')
        self.table_name: Optional[str] = None
        self.columns: Optional[List[str]] = None#注意和之前不一样，对于插入的话数值存储于列表中
        self.values: Optional[List[str]] = None
    
    def to_dict(self):
        return {
            'table_name': self.table_name,
            'columns': self.columns,
            'values': self.values
        }
class parser_sql:
    def __init__(self, sql):
        self.sql = sql
    def parse_sql(self):
        self.sql = self.sql.strip().upper()
        sql_list = self.sql.split(" ")
        if sql_list[0] == 'INSERT':
            return self.parse_insert(self.sql)
        else:
            raise Exception("SQL语句错误，无法识别")
    def parse_insert(self, sql):
        insert_node = Insert_Node()
        
        # 匹配两种INSERT格式：
        # 1. INSERT INTO table_name VALUES (value1, value2, ...)
        # 2. INSERT INTO table_name (column1, column2, ...) VALUES (value1, value2, ...)
        pattern1 = r'INSERT\s+INTO\s+(\w+)\s+VALUES\s*\((.+)\)'
        pattern2 = r'INSERT\s+INTO\s+(\w+)\s*\((.+)\)\s+VALUES\s*\((.+)\)'
        
        match1 = re.match(pattern1, sql, re.IGNORECASE)
        match2 = re.match(pattern2, sql, re.IGNORECASE)
        
        if match2:  # 有列名的格式
            table_name, columns_str, values_str = match2.groups()
            insert_node.table_name = table_name
            insert_node.columns = [col.strip() for col in columns_str.split(',')]
            insert_node.values = [val.strip().strip("'") for val in values_str.split(',')]
        elif match1:  # 没有列名的格式
            table_name, values_str = match1.groups()
            insert_node.table_name = table_name
            insert_node.columns = None  # 没有指定列名
            insert_node.values = [val.strip().strip("'") for val in values_str.split(',')]
        else:
            raise ValueError("无效的INSERT语句格式")
        
        return insert_node

if __name__ == '__main__':
    # 测试用例
    test_cases = [
        "INSERT INTO Websites VALUES ('Google', 'USA', 1000)",
        "INSERT INTO Websites (name, country, visitors) VALUES ('Facebook', 'USA', 2000)",
        "INSERT INTO Students (id, name, age) VALUES (1, 'Alice', 20)"
    ]
    
    for i, sql in enumerate(test_cases):
        print(f"测试用例 {i+1}: {sql}")
        parser = parser_sql(sql)
        result = parser.parse_sql().to_dict()
        print(f"解析结果: {result}\n")
if __name__ == '__main__':
    # 测试用例
    test_cases = [
        "INSERT INTO Websites VALUES ('Google', 'USA', 1000)",
        "INSERT INTO Websites (name, country, visitors) VALUES ('Facebook', 'USA', 2000)",
        "INSERT INTO Students (id, name, age) VALUES (1, 'Alice', 20)"
    ]
    
    for i, sql in enumerate(test_cases):
        print(f"测试用例 {i+1}: {sql}")
        parser = parser_sql(sql)
        result = parser.parse_sql().to_dict()
        print(f"解析结果: {result}\n")