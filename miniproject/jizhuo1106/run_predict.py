import subprocess

# The command you would normally run in the terminal
command_gun = ['python', 'predict_gun.py']
# Run the command and capture the output
result_gun = subprocess.run(command_gun, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# Write the output to a file
with open('result_gun.txt', 'w') as file:
    file.write(result_gun.stdout)

command_knife = ['python', 'predict_knife.py']
result_knife = subprocess.run(command_knife, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open('result_knife.txt', 'w') as file:
    file.write(result_knife.stdout)

command_norm = ['python', 'predict_norm.py']
result_norm = subprocess.run(command_norm, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open('result_norm.txt', 'w') as file:
    file.write(result_norm.stdout)