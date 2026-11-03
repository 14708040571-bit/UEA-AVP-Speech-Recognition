from pathlib import Path
import sounddevice as sd
import soundfile as sf
import time

print("Assessment 001")
print("Dataset setup started")

names = [
    "Aaron",
    "Alex",
    "Ali",
    "Charlie",
    "Jacek",
    "Jack",
    "Jade",
    "Joshua",
    "Harriet",
    "Mason",
    "Nathan",
    "Nicholas",
    "Sam",
    "Samuel",
    "Sean",
    "Toby",
    "Tommy",
    "Xingchen",
    "Ziang",
    "Zihan"
]
print("Number of names:", len(names))

project_folder = Path.home() / "Desktop" / "Assessment001"

train_folder = project_folder / "data" / "train"

print("Training folder:")
print(train_folder)
for name in names:
    name_folder = train_folder / name
    name_folder.mkdir(parents=True, exist_ok=True)
    print("Created:", name_folder)
print("Dataset folders are ready!")

fs = 16000
duration = 2

print()
print("开始正式采集训练数据")
print("每个名字目标：20 条录音")
print()

for test_name in names:

    name_folder = train_folder / test_name

    # 找到这个名字下一个应该使用的编号
    next_number = 1

    while (name_folder / f"{test_name}_{next_number:02d}.wav").exists():
        next_number += 1

    # 如果已经有20条，直接进入下一个名字
    if next_number > 20:
        print()
        print(f"{test_name} 已经有 20 条录音，跳过。")
        continue


    # ==========================================
    # 一直录这个名字，直到达到20条
    # ==========================================

    while next_number <= 20:

        output_file = (
            name_folder
            / f"{test_name}_{next_number:02d}.wav"
        )

        while True:

            print()
            print("--------------------------------")
            print(
                f"准备录制："
                f"{test_name}_{next_number:02d}.wav"
            )
            print(
                f"当前进度："
                f"{next_number}/20"
            )
            print(f"请说：{test_name}")
            print("--------------------------------")

            # 3秒倒计时
            for i in range(3, 0, -1):
                print(i)
                time.sleep(1)

            print("Recording...")

            recording = sd.rec(
                int(duration * fs),
                samplerate=fs,
                channels=1,
                device=3
            )

            sd.wait()

            print("Recording finished!")

            print(
                "Maximum amplitude:",
                recording.max()
            )

            print(
                "Minimum amplitude:",
                recording.min()
            )


            # 播放刚录的声音
            print("Playing recording...")

            sd.play(
                recording,
                fs,
                device=4
            )

            sd.wait()

            print("Playback finished!")


            # 询问是否保存
            choice = input(
                "保存吗？"
                "(y = 保存 / r = 重录 / q = 退出): "
            ).strip().lower()


            if choice == "y":

                sf.write(
                    output_file,
                    recording,
                    fs
                )

                print(
                    f"✓ 已保存："
                    f"{output_file.name}"
                )

                # 保存成功后进入下一编号
                next_number += 1

                break


            elif choice == "r":

                print(
                    "✗ 不保存，"
                    "重新录制这一条。"
                )


            elif choice == "q":

                print(
                    "已退出录音程序。"
                )

                raise SystemExit


            else:

                print(
                    "请输入 y、r 或 q。"
                )


    print()
    print("==============================")
    print(f"✓ {test_name} 已完成 20 条录音")
    print("==============================")