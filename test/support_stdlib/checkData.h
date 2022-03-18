//This file checks data transfer between host and device

#ifndef CHECK_DATA_H
#define CHECK_DATA_H

template <typename T>
bool checkData(T* device_iter, T* host_iter, int N){
  for(int i =0; i<N;++i){
    if(*host_iter != *device_iter)return false;
  }
  return true;
}
template <typename T>
bool checkData(const T* device_iter, const T* host_iter, int N){
  for(int i =0; i<N;++i){
    if(*host_iter != *device_iter)return false;
  }
  return true;
}

#endif
