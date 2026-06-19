import random
from pathlib import Path

def process_split(base_path, split_name, fraction):

    
    # 1. 建立目標資料夾的路徑物件 (利用 / 符號串接路徑，這是 pathlib 的特異功能)
    split_dir = base_path / split_name
    mix_dir = split_dir / "mix"
    s1_dir = split_dir / "s1"
    s2_dir = split_dir / "s2"

    # [教學] 檢查最重要的 mix 目錄是否存在，如果不在就印出警告並提早結束這個函式
    if not mix_dir.exists():
        print(f"警告：找不到路徑 {mix_dir}，跳過 {split_name} 資料集。")
        return

    # 2. 抓取所有 wav 檔案
    # .glob("*.wav") 會找尋該資料夾下所有附檔名為 .wav 的檔案
    # 使用 list() 將結果轉換成 Python 的串列 (List)，方便後續計算數量與抽樣
    mix_files = list(mix_dir.glob("*.wav"))
    total_files = len(mix_files)
    
    if total_files == 0:
        print(f"⚠️ 警告：{mix_dir} 中沒有 .wav 檔案。")
        return

    # 3. 計算要抽取的數量，並進行「隨機抽樣」
    # 用來確保數量是整數。例如 100 * 0.5 = 50.0，轉成整數 50
    sample_size = int(total_files * fraction)
    
    # random.sample 會從 mix_files 中隨機抽出 sample_size 個不重複的檔案
    sampled_mix_files = random.sample(mix_files, sample_size)

    # 4. 定義準備要輸出的三個 .scp 文字檔路徑
    mix_scp = split_dir / "mix.scp"
    s1_scp = split_dir / "s1.scp"
    s2_scp = split_dir / "s2.scp"

    # 5. 開啟檔案並寫入資料
    # [教學] with open(...) 語法可以確保檔案寫入完成後「自動關閉」，不用手動呼叫 close()
    with open(mix_scp, "w", encoding="utf-8") as f_mix, \
         open(s1_scp, "w", encoding="utf-8") as f_s1, \
         open(s2_scp, "w", encoding="utf-8") as f_s2:
        
        # 逐一處理剛剛隨機抽出來的檔案
        for mix_file in sampled_mix_files:
            # [教學] .stem 可以只抓取「檔名」，去掉「.wav」副檔名
            # 例如: "audio_01.wav" -> "audio_01"
            filename = mix_file.stem 
            
            # 利用相同檔名，拼湊出 s1 與 s2 的正確檔案路徑
            s1_file = s1_dir / f"{filename}.wav"
            s2_file = s2_dir / f"{filename}.wav"

            # [教學] 寫入格式為：「檔名 絕對路徑」
            # .resolve() 會把相對路徑轉換成電腦看得懂的「絕對路徑」(例如 C:/.../...)
            # \n 代表換行符號
            f_mix.write(f"{filename} {mix_file.resolve()}\n")
            f_s1.write(f"{filename} {s1_file.resolve()}\n")
            f_s2.write(f"{filename} {s2_file.resolve()}\n")

    print(f"✅ [{split_name.upper()}] 處理完成！總數: {total_files} -> 抽樣後: {sample_size}")

def main():    
    # 1. 給定資料夾根目錄 (請替換成真實路徑)
    base_dir_str = "d:/denoiser-main/final_dataset/final_dataset"
    base_path = Path(base_dir_str)
    
    # 2. 直接給定抽樣比例 (0.5 代表 50%，1.0 代表全取) 
    fraction = 0.5 
    
    # ==========================================
    
    print(f"目標資料集根目錄: {base_path.resolve()}")
    print(f"隨機抽樣比例: {fraction} (約 {fraction*100}%)\n")

    # [教學] 我們有三個子資料夾要處理：訓練集(tr)、驗證集(cv)、測試集(tt)
    # 利用 for 迴圈，一次把這三個資料夾丟進 process_split 函式中處理
    splits = ["tr", "cv", "tt"]
    for split in splits:
        process_split(base_path, split, fraction)

# [教學] 這是 Python 程式的進入點，確保程式被直接執行時才會呼叫 main()
if __name__ == "__main__":
    main()