from typing import Optional, List, Dict, Any
import re
class ASTNode:
    def __init__(self,type):
        self.type=type
    def to_dict(self):
        pass
class Delete_Node(ASTNode):
    def __init__(self):
        super().__init__('DELETE')
        self.table_name:str=None
        self.where_clause:Optional[Condition_where]=None
    def to_dict(self):
        return {
            'table_name':self.table_name,
            'where_clause':self.where_clause.to_dict() if isinstance(self.where_clause,Condition_where) else self.where_clause
        }
class Condition_where(ASTNode):
    def __init__(self,left,right,operate):
        super().__init__('Condition')
        self.left=left#分为左侧条件和右侧条件
        self.right=right
        self.operate=operate
    def to_dict(self):
        return {
            'operate':self.operate,
            'left_condition':self.left.to_dict() if isinstance(self.left,Condition_where) else self.left,
            'right_condition':self.right.to_dict() if isinstance(self.right,Condition_where) else self.right
        }
class parser_sql:
    def __init__(self,sql):
        self.sql=sql
    def parse_sql(self):
        self.sql=self.sql.strip().upper()#去除空格，转换为大写
        sql_list=self.sql.split(" ")#分割字符串，返回列表
        if sql_list[0]=='DELETE':#判断是否为delete语句
            return self.parse_delete(self.sql)
        else:
            raise Exception("SQL语句错误，无法识别")
    def parse_delete(self,sql):
        delete_node=Delete_Node()
        delete_pattern =r'DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.*))?(?:\s*;)?$'
        match=re.match(delete_pattern, sql, re.IGNORECASE)
        #print(match.groups())
        if not match:
            raise ValueError("无效的DELETE语句格式")
        table_name, condition = match.groups()
        delete_node.table_name = table_name
        delete_node.table_name=table_name
        if condition is None:#delete删除整个表，但是不修改表结构，任可插入数据
            delete_node.where_clause="all"
        else:
            delete_node.where_clause=self.parse_where(condition)
        return delete_node
    def parse_where(self,clause):
        if ' AND ' in clause:
            and_index=clause.index(' AND ')
            left=clause[:and_index]
            right=clause[and_index + 5:]
            left_parse=self.parse_single(left)
            right_parse=self.parse_single(right)
            return Condition_where(left_parse,right_parse,'AND')
        elif ' OR ' in clause:
            or_index = clause.index(' OR ')
            left = clause[:or_index]
            right = clause[or_index + 4:]
            left_parse=self.parse_single(left)
            right_parse=self.parse_single(right)
            return Condition_where(left_parse,right_parse,'OR')
        else:
            return self.parse_single(clause)
    def parse_single(self,clause):
        operates=['>=','<=','!=','=','>','<']
        for operate in operates:
            if operate in clause:
                left,right=clause.split(operate)
                return Condition_where(left,right,operate)#直接传入条件
if __name__ == '__main__':
    parse_sql=parser_sql("DELETE FROM Websites WHERE name='Facebook' AND country='USA'")
    parse_sql1=parser_sql("DELETE FROM Websites")
    print(parse_sql.parse_sql().to_dict())
    print(parse_sql1.parse_sql().to_dict())
"""
双条件运行结果
{'table_name': 'WEBSITES', 
'where_clause': {'operate': 'AND',
 'left_condition': {'operate': '=', 'left_condition': 'NAME', 'right_condition': "'FACEBOOK'"}, 
 'right_condition': {'operate': '=', 'left_condition': 'COUNTRY', 'right_condition': "'USA'"}}}
"""    
    

        


        
    