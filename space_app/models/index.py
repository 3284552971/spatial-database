from __future__ import annotations

import sys
import os
from typing import Any,List,Dict
from pathlib import Path
import pickle as pkl

file_path = Path(__file__)
data_path = file_path.parent.parent / 'table_data'
data_path.mkdir(parents=True, exist_ok=True)

class Node:
    def __init__(self, key, value, next=None):
        self.key = key
        self.value = value
        self.next = next

    # 兼容旧代码（历史上误写成 init 而非 __init__）
    def init__(self,key,value,next=None):
        self.key=key
        self.value=value
        self.next=next

class HashIndex:
    def __init__(self):
        self.count = 0
        self.hash_table = []
        self.hash_function = None

    # 兼容旧代码（历史上误写成 init 而非 __init__）
    def init__(self):
        self.count = 0
        self.hash_table = []
        self.hash_function = None

    def init_hash_function(self,length):
        self.hash_function = lambda key: hash(key) % length

    def build_index(self,index_data:List[tuple]):
        """
        :type index_data: List[tuple(key, row_index)]
        :return: None
        :rtype: None
        """
        self.count = 0
        self.hash_table = [None] * int(len(index_data) * 2)
        self.init_hash_function(len(self.hash_table))
        
        for key,row_index in index_data:
                self.insert(key,row_index)
    
    def insert(self,key,row_index):
        """插入键值对"""
        # 允许“空索引”直接增量插入：缺少 hash_table/hash_function 时先初始化
        if not hasattr(self, "hash_table") or self.hash_table is None:
            self.hash_table = []
        if not hasattr(self, "hash_function"):
            self.hash_function = None
        if self.hash_function is None or len(self.hash_table) == 0:
            self.hash_table = [None] * 100
            self.init_hash_function(len(self.hash_table))
        has_index = self.hash_function(key)
        if self.hash_table[has_index] is None:
            self.hash_table[has_index]=Node(key,row_index)
            self.count += 1
        else:
            pre = self.hash_table[has_index]
            if pre.key == key:
                raise KeyError(f"Key {key} already exists in HashIndex.")
            while pre.next:
                pre=pre.next
            pre.next=Node(key,row_index)
        
        if self.count / len(self.hash_table) > 0.8:
            self.__rebuild()

    def __rebuild(self):
        old_table = self.hash_table
        new_size = len(old_table) * 2
        self.hash_table = [None] * new_size
        self.init_hash_function(new_size)
        self.count = 0

        for head in old_table:
            node = head
            while node:
                self.insert(node.key, node.value)
                node = node.next
    
    def update(self, old_key, new_key):
        """
        更新哈希表中的键，由于原表中索引可能会被更新，所以键也要更新
        """
        has_index = self.hash_function(old_key)
        new_has_index = self.hash_function(new_key)
        if not self.hash_table[new_has_index]:
            raise KeyError(f"新键{new_key}已经存在，无法更新索引")
        
        self.hash_table[new_has_index] = self.hash_table[has_index]
        self.hash_table[has_index] = None
        self.hash_table[new_has_index].key = new_key

    def delete(self,key):
        """删除键值对"""
        has_index = self.hash_function(key)
        node = self.hash_table[has_index]
        if node is None:
            raise KeyError(f"Key {key} not found in HashIndex.")
        if node.key == key:
            self.hash_table[has_index] = node.next
            self.count -= 1
            return
        prev = node
        node = node.next
        while node:
            if node.key == key:
                prev.next = node.next
                self.count -= 1
                return
            prev = node
            node = node.next
        raise KeyError(f"Key {key} not found in HashIndex.")
        

    def __getitem__(self,key):
        has_index = self.hash_function(key)
        node = self.hash_table[has_index]
        while node:
            if node.key == key:
                return node.value
            node = node.next
        raise KeyError(f"Key {key} not found in HashIndex.")
    
class BtreeNode:
    def __init__(self, isleaf=True):
        self.isleaf = isleaf
        self.keys = []  # 该结点的键列表 [tuple, ...]
        self.children = []  # 子节点列表 [node, ...] or [row_index, ...]
        self.next = None
        self.parent = None

    # 兼容旧代码（历史上误写成 init 而非 __init__）
    def init__(self,isleaf=True):
        self.isleaf=isleaf
        self.keys=[]#该结点的键列表 [tuple, ...]
        self.children=[]#子节点列表 [node, ...] or [row_index, ...]
        self.next=None
        self.parent=None

class BPlusTreeindex:
    def __init__(self, order=4):
        self.order = order  # 阶数
        self.root = BtreeNode(isleaf=True)
        self.depth = 1
        self.leaf_size = max(int(order), 4)
        self.count = 0

    # 兼容旧代码（历史上误写成 init 而非 __init__）
    def init__(self,order=4):
        self.order=order#阶数
        self.root=BtreeNode(isleaf=True)
        self.depth = 1

    def build_index(self,data_index:List[tuple]):
        """
        args:
            data_index: list[tuple(index_key:tuple(Any), row_index)]
        """
        # print(f"开始为表'{self.table_name}'的列'{self.column_name}'构建B+树索引")
        self.leaf_size = max(self.order, len(data_index) // 10)
        for key_val, row_index in data_index:
            if key_val:
                self.insert(key_val,row_index)#索引从0开始，最后设计的表行号也从0开始吧
                self.count += 1
            else:
                raise ValueError("索引不可为空")
    
    def insert(self,key,row_index):#插入键值对
        leaf=self.find_leaf(self.root,key)#找到在哪个叶子节点插入
        self.insert_leaf(leaf,key,row_index)#插入叶子节点（叶子结点包含数据值）
        if len(leaf.keys) > self.leaf_size:
            self.split_leaf(leaf)
    
    def find_leaf(self,node: BtreeNode,key):
        """二分法加递归找到插入的叶子结点（高效）"""
        #也可以用这个进行主键查询
        current_node = node
        while current_node.isleaf == False:
            left,right=0,len(current_node.keys)-1
            while left<=right:
                mid=(left+right)//2
                if key<current_node.keys[mid][0]:
                    right=mid-1
                else:
                    left=mid+1
            child_index=left
            current_node = current_node.children[child_index]
        return current_node
        # if node.isleaf:
        #     return node#递归终止条件
        # left,right=0,len(node.keys)-1
        # while left<=right:
        #     mid=(left+right)//2
        #     if key<node.keys[mid]:
        #         right=mid-1
        #     else:
        #         left=mid+1
        # child_index=left
        # return self.find_leaf(node.children[child_index],key)

    def insert_leaf(self,node:BtreeNode,key,row_index):
        """"实现在一个结点中查找的二分查找,并且插入键值对"""
        if len(node.keys) == 0:
            node.keys.append(key)
            node.children.append(row_index)
            return#防止死循环
        left=0
        right=len(node.keys)-1
        while left<=right:
            mid=(left+right)//2
            if node.keys[mid] == key:
                raise Exception("主键索引不唯一，报错")
            elif node.keys[mid]<key:
                left=mid+1
            else:
                right=mid-1
        insert_index=left
        node.keys.insert(insert_index,key)
        node.children.insert(insert_index,row_index)#在叶子结点的子节点列表插入行号，相同位置

    def split_leaf(self,node):
        if not node.isleaf:
        # 如果是内部节点，调用内部节点分裂方法
            return self.split_internal_node(node)
        mid_index = len(node.keys)//2
        new_leaf = BtreeNode(isleaf=True)
        # 剪切过程
        new_leaf.keys=node.keys[mid_index:]#新加一个结点
        new_leaf.children=node.children[mid_index:]
        node.keys=node.keys[:mid_index]
        node.children=node.children[:mid_index]
        new_leaf.next = node.next
        node.next=new_leaf
        new_leaf.parent=node.parent
        self.insert_toparent(node,new_leaf.keys[0],new_leaf)#插入中间值
    #当时没注意，B＋树的叶子节点和非叶子节点的分裂方式不同！！
    def split_internal_node(self, node):#非叶子节点分裂
        mid_index = len(node.keys) // 2
        middle_key = node.keys[mid_index]
        new_node = BtreeNode(isleaf=False)
        new_node.keys = node.keys[mid_index + 1:]
        new_node.children = node.children[mid_index + 1:]
        node.keys = node.keys[:mid_index]
        node.children = node.children[:mid_index + 1]
        # 更新子节点的父指针
        for child in new_node.children:
            child.parent = new_node
        new_node.parent = node.parent
        self.insert_toparent(node, middle_key, new_node)
    
    def insert_toparent(self,left_node,key,right_node):
        parent =left_node.parent
        if parent is None:
            new_root=BtreeNode(isleaf=False)
            new_root.keys.append(left_node.keys[0])
            new_root.keys.append(key)
            new_root.children=[left_node,right_node]
            left_node.parent=new_root
            right_node.parent=new_root
            self.root=new_root
            return
        child_index=parent.children.index(left_node)
        parent.keys.insert(child_index + 1,key)
        parent.children.insert(child_index+1,right_node)
        right_node.parent=parent
        if len(parent.keys) > self.order-1:#判断父结点是否需要分裂
            self.split_leaf(parent)

    def __getitem__(self,key):
        leaf = self.find_leaf(self.root,key)
        left=0
        right=len(leaf.keys)-1
        while left<=right:
            mid=(left+right)//2
            if leaf.keys[mid]==key:
                return leaf.children[mid]
            elif leaf.keys[mid]<key:
                left=mid+1
            else:
                right=mid-1
        raise KeyError(f"Key {key} not found in BPlusTreeindex.")

class Index:
    """
    索引方法类
    args:
        table_name: string 表名
        type: string 索引类型 (primary, normal, combo)
        index: list[string] 索引列名列表
        data: list[list] 数据列表
        save_task: list 保存任务列表
    """

    def create_index(self, **kwargs):
        """
        args:
            table_name: string 表名
            type: string 索引类型 (primary, normal)
            index: [string] 索引列名列表
            check: bool 是否进行检查
        algorithm:
            return save to index file task -> [path, DATA_index]
        """
        # 这个逻辑在创建表格的时候使用
        table_name = kwargs.get('table_name')
        type = kwargs.get('type', None)
        index = tuple(sorted(kwargs.get('index')))
        check = kwargs.get('check', True)
        table_dir = kwargs.get("table_dir")
        if table_dir is not None and not isinstance(table_dir, Path):
            table_dir = Path(str(table_dir))
        self.index_check(**kwargs)
        # 目录：优先使用外部传入 table_dir，其次从 Manager.table 推导，最后 fallback 到旧 data_path/table_name
        if table_dir is None:
            try:
                tp = self.table.get(table_name)  # type: ignore[attr-defined]
                if tp:
                    table_dir = Path(str(tp)).parent
            except Exception:
                table_dir = None
        if table_dir is None:
            table_dir = data_path / table_name

        os.makedirs(table_dir, exist_ok=True)
        if type == 'primary':
            # 创建主键索引
            index_example = self.build_Hash_tree()
            with open(table_dir / f'index_{"-".join(index)}.pkl', 'wb') as f:
                pkl.dump(index_example, f)
            self.index.setdefault(table_name, {})
            self.index[table_name][index] = table_dir / f'index_{"-".join(index)}.pkl'
        else:

            index_example = self.build_B_tree()
            with open(table_dir / f'index_{"-".join(index)}.pkl', 'wb') as f:
                pkl.dump(index_example, f)
            self.index.setdefault(table_name, {})
            self.index[table_name][index] = table_dir / f'index_{"-".join(index)}.pkl'
        
    def update_indexes_after_insert(self, **kwargs):
        table_name = kwargs.get('table_name')
        new_row = kwargs.get('new_row')
        data_table = kwargs.get('data_table')
        if table_name not in self.index:
            return
        for index_tuple, index_path in self.index[table_name].items():
            with open(index_path, 'rb') as f:
                index_obj = pkl.load(f)
            # index_tuple: ('colA','colB',...) -> 从 row.data 中取对应位置
            try:
                attrs = getattr(data_table, 'attributes', {}) or {}
                if not isinstance(attrs, dict):
                    continue
                cols = list(index_tuple) if isinstance(index_tuple, (list, tuple)) else []
                if not cols:
                    continue
                index_key = tuple(new_row.data[int(attrs[c])] for c in cols if c in attrs)
                if len(index_key) != len(cols):
                    continue
            except Exception:
                continue

            try:
                if isinstance(index_obj, HashIndex):
                    index_obj.insert(index_key, len(data_table)-1)
                elif isinstance(index_obj, BPlusTreeindex):
                    index_obj.insert(index_key, len(data_table)-1)
                with open(index_path, 'wb') as f:
                    pkl.dump(index_obj, f)
            except Exception:
                # 索引维护失败不应阻断 DML；允许后续重建
                continue

    def build_B_tree(self, index_data: List[tuple] | None = None) -> object:
        b_plus_tree = BPlusTreeindex()
        data = index_data or []
        if data:
            b_plus_tree.build_index(data)
        return b_plus_tree

    def build_Hash_tree(self, index_data: List[tuple] | None = None) -> object:
        hash_index = HashIndex()
        data = index_data or []
        if data:
            hash_index.build_index(data)
        return hash_index

    def index_check(self, **kwargs):
        index = tuple(sorted(kwargs.get('index')))
        table_name = kwargs.get('table_name')
        if kwargs.get('check') is False:
            return
        if table_name not in self.index.keys():
            raise ValueError(f"表 {table_name} 不存在")
        if self.index.get(table_name, None) is None:
            raise ValueError(f"表 {table_name} 上已经存在索引")
        if index in self.index[table_name].keys():
            raise ValueError(f"表 {table_name} 上已经存在索引 {index}")
        
