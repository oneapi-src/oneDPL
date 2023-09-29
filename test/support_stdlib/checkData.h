//This file checks data transfer between host and device

#ifndef CHECK_DATA_H
#define CHECK_DATA_H

template <typename T1, typename T2>
bool
checkData(const T1* device_iter, const T2* host_iter, int N)
{
    for (int i = 0; i < N; ++i)
    {
        if (*host_iter != *device_iter)
            return false;
    }
    return true;
}

#endif
