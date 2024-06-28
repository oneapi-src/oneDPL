#include <oneapi/dpl/execution>
#include "support/utils.h"

#if __has_include(<unistd.h>)
#include <unistd.h>
#endif

int main(){
#ifdef _POSIX_VERSION // defined in <unistd.h>
    [[maybe_unused]] pid_t pid = fork();
#endif
    return TestUtils::done();
}
