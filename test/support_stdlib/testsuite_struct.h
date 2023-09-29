#ifndef _TESTSUITE_STRUCT_H
#define _TESTSUITE_STRUCT_H

struct NoexceptMoveConsClass {
  NoexceptMoveConsClass(NoexceptMoveConsClass &&) noexcept;
  NoexceptMoveConsClass &operator=(NoexceptMoveConsClass &&) = default;
};
struct NoexceptMoveAssignClass {
  NoexceptMoveAssignClass(NoexceptMoveAssignClass &&) = default;
  NoexceptMoveAssignClass &operator=(NoexceptMoveAssignClass &&) noexcept;
};


struct NoexceptMoveConsNoexceptMoveAssignClass {
  NoexceptMoveConsNoexceptMoveAssignClass(
      NoexceptMoveConsNoexceptMoveAssignClass &&) noexcept;

  NoexceptMoveConsNoexceptMoveAssignClass &
  operator=(NoexceptMoveConsNoexceptMoveAssignClass &&) noexcept;
};

#endif
