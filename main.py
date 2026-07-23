import os
import sys
import subprocess

def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    agents_dir = os.path.join(base_dir, "Agents")

    scripts = [
        "Agent 1_Process Planning.py",
        "Agent 2_Process execution.py",
        "Agent 3_Summary.py"
    ]

    print("========================================================")
    print("             EPSpec 自动化分析流程启动")
    print("========================================================")
    print(f"工作目录: {base_dir}")
    print(f"智能体目录: {agents_dir}")
    print("--------------------------------------------------------\n")

    for i, script_name in enumerate(scripts, 1):
        script_path = os.path.join(agents_dir, script_name)

        if not os.path.exists(script_path):
            print(f"[Error] 找不到脚本文件: {script_name}")
            print(f"路径: {script_path}")
            sys.exit(1)

        print(f">>> [Step {i}/{len(scripts)}] 正在启动: {script_name}")
        print("v" * 60)

        try:

            result = subprocess.run(
                [sys.executable, script_name],
                cwd=agents_dir,
                check=False
            )

            print("^" * 60)

            if result.returncode != 0:
                print(f"\n[Error] {script_name} 执行异常 (Exit Code: {result.returncode})")
                print("流程已终止，请检查上述错误信息。")
                sys.exit(result.returncode)
            else:
                print(f">>> {script_name} 执行成功。\n")

        except KeyboardInterrupt:
            print("\n[User] 用户中断了流程。")
            sys.exit(0)
        except Exception as e:
            print(f"\n[Exception] 启动脚本时发生未知错误: {e}")
            sys.exit(1)

    print("========================================================")
    print("             所有智能体任务已执行完毕！")
    print("========================================================")

if __name__ == "__main__":
    main()
