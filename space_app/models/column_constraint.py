import sys
import os
# sys.path.append('/root/workdir/space_database/application/')
from typing import Any,List,Dict
from pathlib import Path

file_path = Path(__file__)
data_path = file_path.parent.parent / 'table_data'
data_path.mkdir(parents=True, exist_ok=True)

class Column_Constraint:
    # TODO : 实现列约束
        
    def not_null(self, column_value:List[Any]) -> bool:
        """非空约束检查"""
        return all(value is not None for value in column_value)
    
    def unique(self, column_value:List[Any]) -> bool:
        """唯一约束检查"""
        return len(column_value) == len(set(column_value))
    
    def check_type(self, column_value:Any, expected_type:type) -> bool:
        """类型检查约束"""
        return isinstance(column_value, expected_type)
    
    def check(self, column_value:List[Any], check_func:callable) -> bool:
        """自定义检查约束"""
        return all(check_func(value) for value in column_value)
    
# check = Column_Constraint()
# print(check.not_null.__name__)
