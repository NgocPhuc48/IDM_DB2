import time
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Cấu trúc dữ liệu mô phỏng logic nén đa tiền tố DB2 ---
class DB2PrefixTree:
    def __init__(self, name):
        self.name = name
        self.segments = []  # Danh sách các nhóm: [{'prefix': str, 'suffixes': set}, ...]
        self.max_segment_size = 16 

    def _get_common_prefix(self, s1, s2):
        n = min(len(s1), len(s2))
        for i in range(n):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1[:n]

    def insert(self, keys):
        self.segments = [] # Reset dữ liệu mỗi lần insert
        sorted_keys = sorted(keys)
        for i in range(0, len(sorted_keys), self.max_segment_size):
            group = sorted_keys[i : i + self.max_segment_size]
            if len(group) > 1:
                common = self._get_common_prefix(group[0], group[-1])
            else:
                common = ""
            
            p_len = len(common)
            suffixes = {k[p_len:] for k in group}
            self.segments.append({
                'prefix': common,
                'suffixes': suffixes
            })

    def search(self, keys):
        for k in keys:
            for seg in self.segments:
                if k.startswith(seg['prefix']):
                    suffix = k[len(seg['prefix']):]
                    if suffix in seg['suffixes']:
                        break
        return True

    def search_range(self, all_sorted_keys, min_indices):
        # Mô phỏng quét vùng (Range Scan)
        for idx in min_indices:
            _ = all_sorted_keys[idx : idx + 100]
        return True

# --- 2. Hàm chạy Benchmark ---
def run_integrated_benchmark():
    # Cấu hình dữ liệu vừa đủ để chạy mượt (100k bản ghi)
    key_numbers = 100000 
    iterations = 3
    range_queries = 100
    
    print(f"--- Bắt đầu Benchmark: {key_numbers} keys, {iterations} lần lặp ---")
    
    # Tạo dữ liệu mẫu có cấu trúc (giúp kỹ thuật nén phát huy tác dụng)
    all_data = [f"comp.sci.ai.deeplearning.2026.project_id_{i:06d}" for i in range(key_numbers)]
    random.shuffle(all_data)
    
    # Chia dữ liệu
    values = all_data[20000:]
    values_warmup = all_data[:20000]
    all_sorted = sorted(all_data)
    min_indices = [random.randint(0, len(all_sorted) - 101) for _ in range(range_queries)]

    # Các cấu trúc so sánh
    structures = [DB2PrefixTree("Btree-Std (No Opt)"), DB2PrefixTree("Btree-DB2-Opt")]
    
    # Giả lập logic Std (không nén) cho đối chứng
    structures[0].insert = lambda keys: setattr(structures[0], 'data', set(keys))
    structures[0].search = lambda keys: [k in structures[0].data for k in keys]

    results_raw = {s.name: {'insert': [], 'search': [], 'range': []} for s in structures}

    for i in range(iterations):
        print(f"Đang chạy lần lặp {i+1}...")
        for tree in structures:
            # 1. Warmup
            tree.insert(values_warmup)
            
            # 2. Measure Insert
            t0 = time.perf_counter()
            tree.insert(values)
            results_raw[tree.name]['insert'].append(time.perf_counter() - t0)
            
            # 3. Measure Search
            t0 = time.perf_counter()
            tree.search(values[:10000]) # Tìm kiếm 10k mẫu để tiết kiệm thời gian chạy
            results_raw[tree.name]['search'].append(time.perf_counter() - t0)
            
            # 4. Measure Range
            t0 = time.perf_counter()
            tree.search_range(all_sorted, min_indices)
            results_raw[tree.name]['range'].append(time.perf_counter() - t0)

    return results_raw, 80000, 10000, range_queries

# --- 3. Hàm vẽ biểu đồ ---
def plot_results(results_raw, n_ins, n_search, n_range):
    names = list(results_raw.keys())
    ops_types = ['insert', 'search', 'range']
    counts = {'insert': n_ins, 'search': n_search, 'range': n_range}
    
    # Tính toán Ops/sec
    ops_per_sec = {op: [] for op in ops_types}
    for name in names:
        for op in ops_types:
            avg_t = statistics.mean(results_raw[name][op])
            ops_per_sec[op].append(counts[op] / avg_t)

    # Thiết lập biểu đồ
    x = np.arange(len(ops_types))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#bdc3c7', '#e74c3c'] # Xám cho Std, Đỏ cho DB2

    for i, name in enumerate(names):
        vals = [ops_per_sec[op][i] for op in ops_types]
        rects = ax.bar(x + i*width, vals, width, label=name, color=colors[i], edgecolor='black')
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height/1e6:.2f}M',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_ylabel('Ops/sec (Triệu thao tác/giây)', fontweight='bold')
    ax.set_title('SO SÁNH HIỆU NĂNG: BTREE TIÊU CHUẨN VS DB2 PREFIX OPTIMIZATION', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([o.upper() for o in ops_types], fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# --- 4. Thực thi ---
if __name__ == "__main__":
    raw_data, n_i, n_s, n_r = run_integrated_benchmark()
    plot_results(raw_data, n_i, n_s, n_r)