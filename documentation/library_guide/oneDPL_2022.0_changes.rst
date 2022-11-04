oneDPL 2022.0 Changes
#####################

This page list the breaking changes that occurred between the 2021.7.1 and 2022.0 releases.
Support for C++11 and C++14 has been discontinued. 

The internal parameter n of discard_block_engine was changed
from a int to a std::size_t type to align its implementation with
the С++ std proposal found at https://cplusplus.github.io/LWG/issue3561.
This change lets you utilize the full range of values for the P and R
template parameters of discard_block_engine.
 
 .. Note::

 Note: You may need to rebuild your code.


* The following algorithms have been removed from dpl namespace include/oneapi/dpl/functional:

  * using ::std::binary_function;
 
  * using ::std::unary_function.

* The following algorithms are deprecated in C++17 and removed C++20:

  * include/oneapi/dpl/functional:

    * using ::std::binary_negate; 
   
    * using ::std::not1;          
   
    * using ::std::not2;          
   
    * using ::std::unary_negate;  
   
  * include/oneapi/dpl/type_traits:

    * using ::std::is_literal_type; 
    
    * using ::std::is_literal_type_v; 
    
    * using ::std::result_of;        
    
    * using ::std::result_of_t;       

