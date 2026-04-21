#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   schema.py
@Time    :   2026/03/06 17:14:45
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""


from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Entity:
    id: str
    name: str
    score: float = 0.0
    type: str = "entity"

    def __str__(self):
        return f"E({self.id}, {self.name}, {self.score})"


@dataclass
class Relation:
    id: str
    score: float = 0.0
    left: bool = True
    left_entity: Optional[Any] = field(default=None, init=False)
    right_entity: Optional[Any] = field(default=None, init=False)
    
    def set_entity(self, entity, left=True):
        if left:
            self.left_entity = entity
        else:
            self.right_entity = entity

    def __str__(self):
        if self.left:
            return f"{self.left_entity} -- R({self.id}, {self.score}) --> {self.right_entity}"
        else:
            return f"{self.left_entity} <-- R({self.id}, {self.score}) -- {self.right_entity}"