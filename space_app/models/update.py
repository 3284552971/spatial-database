from typing import Optional, List, Dict, Any
import re

class ASTNode:
    def __init__(self, node_type):
        self.node_type = node_type
    
    def to_dict(self):
        pass

class Condition_where(ASTNode):
    def __init__(self, left, right, operate):
        super().__init__('Condition')
        self.left = left
        self.right = right
        self.operate = operate
    
    def to_dict(self):
        return {
            'operate': self.operate,
            'left_condition': self.left.to_dict() if isinstance(self.left, Condition_where) else self.left,
            'right_condition': self.right.to_dict() if isinstance(self.right, Condition_where) else self.right
        }

class Update_Node(ASTNode):
    def __init__(self):
        super().__init__('UPDATE')
        self.table_name: Optional[str] = None
        self.set_clause: Optional[Dict[str, str]] = None
        self.where_clause: Optional[Condition_where] = None
    
    def to_dict(self):
        return {
            'table_name': self.table_name,
            'set_clause': self.set_clause,
            'where_clause': self.where_clause.to_dict() if isinstance(self.where_clause, Condition_where) else self.where_clause
        }

class parser_sql:
    def __init__(self, sql):
        self.sql = sql
    
    def parse_sql(self):
        self.sql = self.sql.strip().upper()
        sql_list = self.sql.split(" ")
        if sql_list[0] == 'UPDATE':
            return self.parse_update(self.sql)
        else:
            raise Exception("SQL语句错误，无法识别")
    
    def parse_update(self, sql):
        update_node = Update_Node()
        
        # 匹配UPDATE格式：UPDATE table_name SET column1=value1, column2=value2 WHERE condition
        pattern = r'UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?$'
        
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            raise ValueError("无效的UPDATE语句格式")
        
        table_name, set_clause, where_condition = match.groups()
        update_node.table_name = table_name
        
        # 解析SET子句
        set_pairs = [pair.strip() for pair in set_clause.split(',')]
        update_node.set_clause = {}
        for pair in set_pairs:
            if '=' in pair:
                column, value = pair.split('=', 1)
                update_node.set_clause[column.strip()] = value.strip().strip("'")
        
        # 解析WHERE条件
        if where_condition:
            update_node.where_clause = self.parse_where(where_condition.strip())
        else:
            update_node.where_clause = None
        
        return update_node
    
    def parse_where(self, clause):
        if ' AND ' in clause:
            and_index = clause.index(' AND ')
            left = clause[:and_index].strip()
            right = clause[and_index + 5:].strip()
            left_parse = self.parse_single(left)
            right_parse = self.parse_single(right)
            return Condition_where(left_parse, right_parse, 'AND')
        elif ' OR ' in clause:
            or_index = clause.index(' OR ')
            left = clause[:or_index].strip()
            right = clause[or_index + 4:].strip()
            left_parse = self.parse_single(left)
            right_parse = self.parse_single(right)
            return Condition_where(left_parse, right_parse, 'OR')
        else:
            return self.parse_single(clause)
    
    def parse_single(self, clause):
        operates = ['>=', '<=', '!=', '=', '>', '<']
        for operate in operates:
            if operate in clause:
                left, right = clause.split(operate, 1)
                return Condition_where(left.strip(), right.strip().strip("'"), operate)

if __name__ == '__main__':
    # 测试用例
    test_cases = [
        "UPDATE Websites SET name='Google' WHERE id=1",
        "UPDATE Websites SET visitors=1000, country='USA' WHERE name='Google' AND id=1",
        "UPDATE Students SET age=21, grade='A' WHERE name='Alice'"
    ]
    
    for i, sql in enumerate(test_cases):
        print(f"测试用例 {i+1}: {sql}")
        parser = parser_sql(sql)
        result = parser.parse_sql().to_dict()
        print(f"解析结果: {result}\n")
"""
{'table_name': 'STUDENTS', 
'set_clause': {'AGE': '21', 'GRADE': 'A'}, 
'where_clause': {'operate': '=', 'left_condition': 'NAME', 'right_condition': 'ALICE'}}

"""