# @Author: Alfonsas Jursenas
# @Date:   2020-04-19T13:33:52+03:00
# @Email:  alfonsas.jursenas@gmail.com
# @Last modified by:   Alfonsas Jursenas
# @Last modified time: 2020-04-19T15:55:42+03:00



from __future__ import absolute_import
from .differential_operators import (
    make_vjp, grad, multigrad_dict,
    elementwise_grad,
    elementwise_holomorphic_grad,
     elementwise_grad_c, value_and_grad,
    grad_and_aux, hessian_tensor_product, hessian_vector_product, hessian,
    jacobian, tensor_jacobian_product, vector_jacobian_product, grad_named,
    checkpoint, make_hvp, make_jvp, make_ggnvp, deriv, holomorphic_grad)
from .builtins import isinstance, type, tuple, list, dict
from autograd.core import primitive_with_deprecation_warnings as primitive
