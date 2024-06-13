import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Set

import abc


@dataclass
class Frames:
    frames: List


# dataclasses (или named­tup­les), — это эквивалентность значений, причудливый способ сказать, что
# «две позиции заказа с одинаковыми orderid, sku и qty идентичны
@dataclass(frozen=True)
class PixelCoordinate:
    height: int
    weight: int

@dataclass
class XYCoordinate:
    " Meters or santimetrs "
    x: float
    y: float


@dataclass
class RectanglePoints:
    # Rectangle
    x1: int = None
    y1: int = None
    x2: int = None
    y2: int = None


@dataclass(frozen=True)
class PoseKeyPoints:
    # point_1: PixelCoordinate
    # point_1: List[int, int]
    # point_n: List[int, int]
    pass


# @dataclass(frozen=True)
# class CourtKeyPoints:
#     keypoints: List[KeyPoint]



# class Model:
#     def __init__(self, path_2_model) -> None:
#         self.path = path_2_model
#         # 
#         self.model = None

#     def read_model(self):
#         """  """
#         # Load the model
#         # model = 

#         # assert model != None
#         # return model
#         pass
    
#     def predict(self, X):
#         """ """
#         predict = self.model(X)
#         predict = self.postprocess(predict)

#         assert len(X) == len(predict)
#         return predict 
        
#     def postrocess(self, predict):
#         """  """
#         # For istance, convert to numpy array and so on...
#         return predict

#     pass  # А нужно ли это нам????
 

# class Detectors(abc.ABC):
#     @abc.abstractmethod
#     def add(self, batch: model.Batch):
#         raise NotImplementedError
    
#     @abc.abstractmethod
#     def get(self, reference) -> model.Batch:
#         raise NotImplementedError     


# # dataclasses (или named­tup­les), — это эквивалентность значений, причудливый способ сказать, что
# # «две позиции заказа с одинаковыми orderid, sku и qty идентичны
# @dataclass(frozen=True)
# class OrderLine:
#     orderid: str
#     sku: str
#     qty: int
    

# class Batch:
#     """
#     OrderLine — это немутируемый класс данных без какого-либо поведения1
#     """
#     def __init__(
#         self, ref: str, sku: str, qty: int, eta: Optional[date]
#     ):
#         self.reference = ref
#         self.sku = sku
#         self.eta = eta
#         self.available_quantity = qty

#     def allocate(self, line: OrderLine):
#         self.available_quantity -= line.qty

#     def deallocate(self, line: OrderLine):
#         if line in self._allocations:
#             self._allocations.remove(line)

#     @property
#     def allocated_quantity(self) -> int:
#         return sum(line.qty for line in self._allocations)
    
#     @property
#     def available_quantity(self) -> int:
#         return self._purchased_quantity - self.allocated_quantity
    
#     def can_allocate(self, line: OrderLine) -> bool:
#         return self.sku == line.sku and self.available_quantity >= line.qty
    

# def test_allocating_to_a_batch_reduces_the_available_quantity():
#     batch = Batch("batch-001", "SMALL-TABLE", qty=20, eta=date.today())
#     line = OrderLine('order-ref', "SMALL-TABLE", 2)
#     batch.allocate(line)
#     assert batch.available_quantity == 18
#     return True


# class AbstractRepository(abc.ABC):
#     @abc.abstractmethod
#     def add(self, batch: model.Batch):
#         raise NotImplementedError
#     @abc.abstractmethod
#     def get(self, reference) -> model.Batch:
#         raise NotImplementedError 
    

# class OutOfStock(Exception):
#     pass


# def allocate(line: OrderLine, batches: List[Batch]) -> str:
#     try:
#         batch = next(b for b in sorted(batches) if b.can_allocate(line))
#         batch.allocate(line)
#         return batch.reference
#     except StopIteration:
#         raise OutOfStock(f"Out of stock for sku {line.sku}")