python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/temp_kernel_linear --repeat=1 --n1s=1 --n2s=1
python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/temp_kernel_quadratic --repeat=1 --n1s=1 --n2s=1
python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/temp_kernel_gaussian --repeat=1 --n1s=1 --n2s=1


python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear --repeat=50 --n1s=5 --n2s=6
python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic --repeat=50 --n1s=5 --n2s=6
python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian --repeat=50 --n1s=5 --n2s=6
