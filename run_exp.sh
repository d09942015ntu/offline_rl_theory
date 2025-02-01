
#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/temp_kernel_linear --repeat=1 --n1s=1 --n2s=1
#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/temp_kernel_quadratic --repeat=1 --n1s=1 --n2s=1
#python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/temp_kernel_gaussian --repeat=1 --n1s=1 --n2s=1
#
#
#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear --repeat=50 --n1s=5 --n2s=6
#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic --repeat=50 --n1s=5 --n2s=6
#python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian --repeat=50 --n1s=5 --n2s=6

#python3 evaluator_fzlake.py --kernel=kernel_linear --phi=phi_tabular --output_dir=results/temp_fzlake_tabular --repeat=1 --n1s=1 --n2s=1
#python3 evaluator_fzlake.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/temp_fzlake_gaussian --repeat=1 --n1s=1 --n2s=1
#
#python3 evaluator_fzlake.py --kernel=kernel_linear --phi=phi_tabular --output_dir=results/temp2_fzlake_tabular --repeat=5 --n1s=5 --n2s=6
#python3 evaluator_fzlake.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/temp2_fzlake_gaussian --repeat=5 --n1s=5 --n2s=6

#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear --repeat=50 --n1s=5 --n2s=6
#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic --repeat=50 --n1s=5 --n2s=6
#python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian --repeat=50 --n1s=5 --n2s=6


n2ss=(10 20 50 100 200 500)
for n2s in ${n2ss[@]}; do
  python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_cubic --output_dir=results/kernel_cubic_100 --repeat=50 --n1s 100 --n2s ${n2s} &
  #python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic_100 --repeat=50 --n1s 100  --n2s ${n2s} &
  #python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear_100 --repeat=50 --n1s 100  --n2s ${n2s} &
  #python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian_100 --repeat=50 --n1s 100  --n2s ${n2s} &
  python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_cubic --output_dir=results/kernel_cubic_50 --repeat=50 --n1s 50  --n2s ${n2s} &
  #python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic_50 --repeat=50 --n1s 50  --n2s ${n2s} &
  #python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear_50 --repeat=50 --n1s 50  --n2s ${n2s} &
  #python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian_50 --repeat=50 --n1s 50  --n2s ${n2s} &
done
