#include <oneapi/dpl/execution>
#include "support/utils.h"

#ifndef _WIN32
#include <unistd.h>
#endif

int main(){
#ifndef _WIN32
    [[maybe_unused]] pid_t pid = fork();
#endif
    return TestUtils::done();
}
