////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source
// License.  See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//
// File created by:
// Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
////////////////////////////////////////////////////////////////////////////////

/** @file
 *  @brief implementation of openmp specialization of TasksOneToOne
 */
#ifndef QMCPLUSPLUS_TASKSONETOONEOPENMP_HPP
#define QMCPLUSPLUS_TASKSONETOONEOPENMP_HPP

#include <stdexcept>
#include <string>
#include <omp.h>

#include "Concurrency/TasksOneToOne.hpp"
#include "Utilities/OutputManager.h"

namespace qmcplusplus
{
/** implements task per thread launch for openmp.
 *
 *  This specialization throws below the top openmp theading level
 *  exception must be caught at thread level or terminate is called.
 *
 */
template<>
template<typename F, typename... Args>
void TasksOneToOne<Threading::OPENMP>::operator()(F&& f, Args&&... args)
{
  const std::string nesting_error{"TasksOneToOne should not be used for nested openmp threading\n"};
  if (omp_get_level() > 0)
      throw std::runtime_error(nesting_error);
  int nested_throw_count = 0;
  int throw_count = 0;
#pragma omp parallel num_threads(num_threads_) reduction(+ : nested_throw_count, throw_count)
  {
    try
    {
      f(omp_get_thread_num(), std::forward<Args>(args)...);
    }
    catch (const std::runtime_error& re)
    {
      if(nesting_error == re.what())
          ++nested_throw_count;
      else
      {
        app_warning() << re.what();
        ++throw_count;
      }
    }
    catch (...)
    {
      ++throw_count;
    }
  }
  if (throw_count > 0)
    throw std::runtime_error("Unexpected exception thrown in threaded section");
  else if (nested_throw_count > 0)
    throw std::runtime_error(nesting_error);
}

} // namespace qmcplusplus

#endif
