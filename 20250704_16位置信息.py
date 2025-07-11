from ultralytics import settings

# View all settings
print(settings)

# Return a specific setting
value = settings["runs_dir"]

print(value)
# value 变量的内容 ，是一个与 settings 中的 "runs_dir" 设置相关的路径。