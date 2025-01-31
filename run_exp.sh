
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

python3 evaluator_fzlake.py --kernel=kernel_linear --phi=phi_tabular --output_dir=results/fzlake_tabular_200 --repeat=20 --n1s=200
python3 evaluator_fzlake.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/fzlake_gaussian_200 --repeat=20 --n1s=200

python3 evaluator_fzlake.py --kernel=kernel_linear --phi=phi_tabular --output_dir=results/fzlake_tabular_500 --repeat=20 --n1s=500
python3 evaluator_fzlake.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/fzlake_gaussian_500 --repeat=20 --n1s=500

python3 evaluator_fzlake.py --kernel=kernel_linear --phi=phi_tabular --output_dir=results/fzlake_tabular_1000 --repeat=20 --n1s=1000
python3 evaluator_fzlake.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/fzlake_gaussian_1000 --repeat=20 --n1s=1000

#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_linear --output_dir=results/kernel_linear --repeat=50 --n1s=5 --n2s=6
#python3 evaluator_carpole.py --kernel=kernel_linear --phi=phi_quadratic --output_dir=results/kernel_quadratic --repeat=50 --n1s=5 --n2s=6
#python3 evaluator_carpole.py --kernel=kernel_gaussian --phi=phi_array --output_dir=results/kernel_gaussian --repeat=50 --n1s=5 --n2s=6
