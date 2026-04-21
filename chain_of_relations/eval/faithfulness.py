#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    :   faithfulness.py
@Time    :   2026/04/06 15:17:07
@Author  :   liuchenhui
@Contact :   liuchh9@mail2.sysu.edu.cn
@Desc    :   None
"""

def update_knowledge_distribution(knowledge_distribution: dict, predict_type: str):
    """Update action distribution used for KGR-style faithfulness reporting."""
    if predict_type not in knowledge_distribution:
        knowledge_distribution[predict_type] = 0
    knowledge_distribution[predict_type] += 1


def sorted_knowledge_distribution(knowledge_distribution: dict):
    """Sort action distribution by frequency in descending order."""
    return sorted(knowledge_distribution.items(), key=lambda x: x[1], reverse=True)

