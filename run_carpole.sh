


n1ss=(20 50 100 200)
for n1s in ${n1ss[@]}; do
  python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear_${n1s} --repeat=50 --n1s ${n1s} &
  python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic_${n1s} --repeat=50 --n1s ${n1s} &
  python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_cubic --output_dir=results/kernel_cubic_${n1s} --repeat=50 --n1s ${n1s} &
  python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian_${n1s} --repeat=50   --n1s ${n1s} &
done
