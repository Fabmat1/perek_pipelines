subroutine make_bins(wavs, edges, widths, n)
  implicit none
  integer, intent(in) :: n
  real*8, intent(in) :: wavs(n)
  real*8, intent(out) :: edges(n+1)
  real*8, intent(out) :: widths(n)

  integer :: i

  edges(1) = wavs(1) - (wavs(2) - wavs(1))/2.0
  widths(n) = wavs(n) - wavs(n-1)
  edges(n+1) = wavs(n) + (wavs(n) - wavs(n-1))/2.0
  do i = 2, n
    edges(i) = (wavs(i) + wavs(i-1)) / 2.0
  end do
  do i = 1, n-1
    widths(i) = edges(i+1) - edges(i)
  end do

end subroutine make_bins

subroutine resample(x, y, nx, xx, yy, nxx, fill)

!f2py real*8, intent(in) :: x
!f2py real*8, intent(in) :: y
!f2py integer, intent(in) :: nx
!f2py real*8, intent(in) :: xx
!f2py real*8, intent(inout) :: yy
!f2py integer, intent(in) :: nxx
!f2py real*8, intent(in) :: fill
!f2py depend(nx) x
!f2py depend(nx) y
!f2py depend(nxx) xx
!f2py depend(nxx) yy

  implicit none

  integer, intent(in) :: nx
  real*8, intent(in) :: x(nx)
  real*8, intent(in) :: y(nx)
  integer, intent(in) :: nxx
  real*8, intent(in) :: xx(nxx)
  real*8, intent(out) :: yy(nxx)
  real*8, intent(in) :: fill

  real*8 :: old_edges(nx+1), old_widths(nx)
  real*8 :: new_edges(nxx+1), new_widths(nxx)
  real*8 :: start_factor, end_factor, f_widths_sum
  integer :: start, stop, j, i

  call make_bins(x, old_edges, old_widths, nx)
  call make_bins(xx, new_edges, new_widths, nxx)

  start = 1
  stop = 1

  do j = 1, nxx
    if (new_edges(j) < old_edges(1) .or. new_edges(j+1) > old_edges(nx+1)) then
      yy(j) = fill
      cycle
    endif

    do while (old_edges(start+1) <= new_edges(j))
      start = start + 1
    end do

    do while (old_edges(stop+1) < new_edges(j+1))
      stop = stop + 1
    end do

    if (stop == start) then
      yy(j) = y(start)
    else
      start_factor = (old_edges(start+1) - new_edges(j)) / (old_edges(start+1) - old_edges(start))
      end_factor = (new_edges(j+1) - old_edges(stop)) / (old_edges(stop+1) - old_edges(stop))

      old_widths(start) = old_widths(start) * start_factor
      old_widths(stop) = old_widths(stop) * end_factor

      f_widths_sum = 0.0
      yy(j) = 0.0
      do i = start, stop
        yy(j) = yy(j) + old_widths(i) * y(i)
        f_widths_sum = f_widths_sum + old_widths(i)
      end do

      yy(j) = yy(j) / f_widths_sum

      old_widths(start) = old_widths(start) / start_factor
      old_widths(stop) = old_widths(stop) / end_factor
    endif
  end do

end subroutine resample

