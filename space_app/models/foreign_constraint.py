import sys
import os
# sys.path.append('/root/workdir/space_database/application/')
from typing import Any,List,Dict
from pathlib import Path
import pickle as pkl

file_path = Path(__file__)
data_path = file_path.parent.parent / 'table_data'
data_path.mkdir(parents=True, exist_ok=True)

class Foreign_Constraint:
    # TODO : 实现外键约束
    def create(self, **kwargs):
        """
        创建外键约束
        args:
            
        """
        foreign_table = kwargs.get('foreign_table')
        foreign_column = kwargs.get('foreign_column')
        local_column = kwargs.get('local_column')
        table_name = kwargs.get('table_name')
        key = (table_name, foreign_table, local_column, foreign_column)
        if key in self.foreign_key:
            raise ValueError(f"外键约束 {key} 已经存在")
        self.foreign_key[key] = True

    def check_foreign_key_target(self, foreign_table:str, foreign_column:str, ROW, table_name, local_column:str):
        """
        检查外键约束是否满足，针对单个外键约束
        args:
            foreign_table: 外键表名
            foreign_column: 外键列名
            local_column: 本地列名
            table_name: 当前表名
            attr: 当前表属性字典
        """
        check_exists = self.foreign_key[tuple([table_name, foreign_table, local_column, foreign_column])]
        if not check_exists:
            raise ValueError(f"外键约束 {table_name}.{local_column} -> {foreign_table}.{foreign_column} 不存在")
        # 加载外键表数据
        foreign_table_path = data_path / foreign_table / 'data_table.pkl'
        if not foreign_table_path or not foreign_table_path.exists():
            raise ValueError(f"外键表 {foreign_table} 不存在")
        with open(foreign_table_path, 'rb') as f:
            foreign_data_table = pkl.load(f)
        if foreign_column not in foreign_data_table.attrs:
            raise ValueError(f"外键列 {foreign_column} 不存在于表 {foreign_table} 中")
        foreign_col_index = foreign_data_table.attrs[foreign_column]
        foreign_values = set()
        for row in foreign_data_table.data:
            foreign_values.add(row[foreign_col_index])
        
        # 检查外键约束
        if ROW.local_column not in foreign_values:
            raise ValueError(f"外键约束违反: 值 {ROW.local_column} 在表 {foreign_table} 的列 {foreign_column} 中不存在")
        

    def check_foreign_key(self, ROW, table_name):
        """
        检查外键约束是否满足
        args:
            ROW: 当前行数据
            table_name: 当前表名
            attr: 当前表属性字典
        """
        for key in self.foreign_key.keys():
            fk_table, fk_foreign_table, fk_local_column, fk_foreign_column = key
            if fk_table == table_name:
                self.check_foreign_key_target(
                    foreign_table=fk_foreign_table,
                    foreign_column=fk_foreign_column,
                    ROW=ROW,
                    table_name=table_name,
                    local_column=fk_local_column
                )
