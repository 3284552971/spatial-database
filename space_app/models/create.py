import re
from typing import Dict,List,Any,Optional,Union
from abc import abstractmethod
class ASTNode:
    def __init__(self,type):
        self.type=type
    @abstractmethod
    def to_dict(self):
        pass
class Create_Node(ASTNode):
    def __init__(self):
        super().__init__('CREATE')
        self.table_name:str= None
        self.columns:Optional[colunms_Node] = None
        self.columns_condition:Dict={}
    def to_dict(self):
        return {
                "table_name":self.table_name,
                "columns_condition":self.columns_condition,#存储列的属性，例如主键属性或者不为空等属性
                "column_info":self.columns.to_dict() if self.columns else None,#防止报错，存储创建列的名称和创建列的数据类型，还有主键列

            }
class colunms_Node(ASTNode):
    def __init__(self):
        super().__init__('COLUMNS')
        self.column_name_type:Dict={}#存储列名和类型
        self.primary_key=None
    def to_dict(self):
        return {
            'column_name':self.column_name_type,#判断列类型，例如可变长字符串等
            'primary_key':self.primary_key#主键
        }
class parser_sql:
    @staticmethod
    def parse(sql):
        sql=sql.upper()#正则表达式解析
        create_pattern = r'CREATE\s+TABLE\s+(\w+)\s*\((.*)\)'
        match = re.match(create_pattern, sql, re.IGNORECASE)
        table_name,condition=match.groups() # type: ignore
        create_Node=Create_Node()
        create_Node.table_name=table_name
        condition=condition.split(",")
        condition_list=[]
        for i in range(len(condition)):
            condition_list.append(condition[i].split(" ",2))
        for i in condition_list:
            create_Node.columns_condition[i[0]]=i[-1]
        create_Node.columns=parser_sql.parser_columns(condition_list)
        return create_Node
    @staticmethod
    def parser_columns(condition_list):
        colunms_node=colunms_Node()
        for i in condition_list:
            colunms_node.column_name_type[i[0]]=i[1]#数据类型
            if i[-1]=="PRIMARY KEY":
                colunms_node.primary_key=i[0]
        return colunms_node
if __name__=="__main__":
    parser=parser_sql()
    print(parser.parse("CREATE TABLE Students (id INT PRIMARY KEY,name VARCHAR(50) NOT NULL)").to_dict())
#{'table_name': 'STUDENTS', 
# 'columns_condition': {'ID': 'PRIMARY KEY', 'NAME': 'NOT NULL'}, 
# 'column_info': {'column_name_type': {'ID': 'INT', 'NAME': 'VARCHAR(50)'},
#  'primary_key': 'ID'}}
            

        
        