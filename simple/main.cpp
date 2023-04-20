#include <iostream>
#include "scheduler.hpp"

Task TaskA(Scheduler& sched) {
  std::cout << "Hello from TaskA\n";
  co_await sched.suspend();
  std::cout << "Executing the TaskA\n";
  co_await sched.suspend();
  std::cout << "TaskA is finished\n";
}

Task TaskB(Scheduler& sched) {
  std::cout << "Hello from TaskB\n";
  co_await sched.suspend();
  std::cout << "Executing the TaskB\n";
  co_await sched.suspend();
  std::cout << "TaskB is finished\n";
}


int main() {

  Scheduler sched;

  TaskA(sched);
  TaskB(sched);

  while(sched.schedule()) {};
}


