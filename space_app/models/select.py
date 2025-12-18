from __future__ import annotations

import re
from typing import Dict,List,Any,Optional,Union
from pathlib import Path
from datetime import datetime
from abc import ABC,abstractmethod
class ASTNode:#语法树结点
    def __init__(self,node_type:str):
        self.node_type=node_type
    @abstractmethod#修饰为抽象方法，子类必须实现
    def to_dict(self):#将每一个语法结点转为字典表达
        """将节点转为字典表达"""
        raise NotImplementedError("子类必须实现to_dict方法")#抛出异常
class SelectNode(ASTNode):#查询
    def __init__(self):
        super().__init__("SELECT")  # 调用父类的初始化方法,初始化结点为select结点
        self.columns: Optional[ColumnNode] = None  # 修复：使用 Optional 允许 None
        self.from_table: Optional[str] = None  
        self.where: Optional[Condition_where] = None 
        self.order_by: Optional[Order_Node] = None  
        self.limit: Optional[Limit] = None  
        #py的重写不需要保持参数一致
    def to_dict(self):
        return {
            'node_type':self.node_type,
            'columns':self.columns.to_dict() if self.columns else None,
            'table_name':self.from_table,
            'where_conditon':self.where.to_dict() if self.where else None,
            'order_by':self.order_by.to_dict() if self.order_by else None,
            'limit':self.limit.to_dict() if self.limit else None#待实现
        }

class ColumnNode(ASTNode):
    def __init__(self,name:list[str]):
        super().__init__('COLUMN')#定义结点类型
        self.column_name=name
        self.count=0
    def to_dict(self):
        return {
            'column_name':self.column_name,
            'count':len(self.column_name)
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
class Order_Node(ASTNode):
    def __init__(self,column,direction):
        super().__init__('ORDER')
        self.column=column
        self.direction=direction
    def to_dict(self):
        return {
            'column_name':self.column,
            'direction':self.direction
        }
class Limit(ASTNode):
    def __init__(self,limit):
        super().__init__('Limit')
        self.limit=limit
    def to_dict(self):
        return {
        'max_row':self.limit
        }
class SQL_parser:
    def __init__(self,sql):
        self.keyword_DDL = ['CREATE', 'DROP', 'ALTER']
        self.keyword_DML = ['INSERT', 'DELETE', 'UPDATE']
        self.keyword_DQL = ['FROM', 'WHERE', 'GROUP BY', 'SELECT', 'ORDER BY']
        self.sql=sql
    def parser_sql(self):
        self.sql=self.sql.strip().upper()
        sql_list=self.sql.split(" ")
        if sql_list[0] == 'SELECT':
            return self.parse_select(self.sql)
        else:
            raise Exception("SQL语句错误")
    def parse_select(self,sql):#解析后存储在语法树中
        select_node = SelectNode()
        pattern = r'SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+?))?(?:\s+ORDER\s+BY\s+(.+?))?(?:\s+LIMIT\s+(\d+))?$'
        match = re.match(pattern,sql,re.IGNORECASE)#忽略空格，匹配并打包匹配对象
        if match is None:
            print("错误的sql语句")
        else:
            columns_name, table_name, where_clause, order_by, limit = match.groups()
        select_node.from_table=table_name
        select_node.columns=ColumnNode(columns_name.split(","))#分裂字符串，可以直接使用该结点
        if where_clause:
            select_node.where=self.parse_where(where_clause)
        if order_by:
            select_node.order_by=self.parse_order_by(order_by)
        if limit:
            select_node.limit=Limit(limit)#最大查询语句个数
        return select_node#赋值属性后返回语法树的select结点
    def parse_where(self, clause):# 解析where条件
        clause=clause.split(" ")
        if 'AND' in clause:
            Condition_and=clause.index('AND')
            left=clause[Condition_and-1]
            right=clause[Condition_and+1]
            left_parse=self.parse_single(left)
            right_parse=self.parse_single(right)
            return Condition_where(left_parse,right_parse,'AND')
        elif 'OR' in clause:
            Condition_or=clause.index('OR')
            left=clause[Condition_or-1]
            right=clause[Condition_or+1]
            left_parse=self.parse_single(left)
            right_parse=self.parse_single(right)
            return Condition_where(left_parse,right_parse,'OR')
        else:
            return self.parse_single(clause[0])#调用解析单个条件的方法
    def parse_single(self,clause):
        operates=['>=','<=','!=','=','>','<']
        for operate in operates:
            if operate in clause:
                left,right=clause.split(operate)
                return Condition_where(left,right,operate)#直接传入条件
    def parse_order_by(self,clause):
        attr=clause.split(" ")
        order_Node=Order_Node(attr[0],attr[1])
        return order_Node
if __name__ == "__main__":
    parser = SQL_parser("select a from b where a>1 and c<3 order by d desc limit 10")
    print(parser.parser_sql().to_dict())
#{'node_type': 'SELECT',
#  'columns': {'column_name': ['A'], 'count': 1},
#  'table_name': 'B', 'where_conditon': {'operate': 'AND', 'left_condition': {'operate': '>', 'left_condition': 'A', 'right_condition': '1'}, 'right_condition': {'operate': '<', 'left_condition': 'C', 'right_condition': '3'}}, 
# 'order_by': {'column_name': 'D', 'direction': 'DESC'},
#  'limit': {'max_row': '10'}}

    

        
        

            
            
            
        



    
        
