#ifndef _TESTSUITE_STRUCT_H
#define _TESTSUITE_STRUCT_H

struct NoexceptMoveConsClass {
  NoexceptMoveConsClass(NoexceptMoveConsClass &&) noexcept(true);
  NoexceptMoveConsClass &operator=(NoexceptMoveConsClass &&) = default;
};
struct NoexceptMoveAssignClass {
  NoexceptMoveAssignClass(NoexceptMoveAssignClass &&) = default;
  NoexceptMoveAssignClass &operator=(NoexceptMoveAssignClass &&) noexcept(true);
};


struct NoexceptMoveConsNoexceptMoveAssignClass {
  NoexceptMoveConsNoexceptMoveAssignClass(
      NoexceptMoveConsNoexceptMoveAssignClass &&) noexcept(true);

  NoexceptMoveConsNoexceptMoveAssignClass &
  operator=(NoexceptMoveConsNoexceptMoveAssignClass &&) noexcept(true);
};

#endif
