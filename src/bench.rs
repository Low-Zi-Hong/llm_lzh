// 1. 定义你想监控的算子清单
#[derive(Debug, Clone, Copy)]
pub enum FnIndex {
    TokenEmbedding = 0,
    RmsNormInput   = 1,
    QkvProj        = 2,
    rope           = 8,
    AttentionScore = 3,
    Softmax        = 4,
    OutProj        = 5, // 今晚的主战场
    MlpBlock       = 6,
    LmHead         = 7,
}

pub struct GlobalMonitor {
    // 你把大小改成了 9，说明你加了新算子，没问题！
    pub accum_times: [std::time::Duration; 9], 
    pub start_slot: std::time::Instant,
}

impl GlobalMonitor {
    pub fn new() -> Self {
        Self {
            // 【修正 1】：不能写类型，必须用 Duration::ZERO 填充初始值
            accum_times: [std::time::Duration::ZERO; 9],
            start_slot: std::time::Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        // 【修正 2】：不能直接赋 0，要用 Duration::ZERO
        self.accum_times = [std::time::Duration::ZERO; 9];
    }

    #[inline(always)]
    pub fn enter(&mut self) {
        self.start_slot = std::time::Instant::now();
    }

    #[inline(always)]
    pub fn exit(&mut self, func: FnIndex) {
        // 【极致优雅】：Duration 原生支持 += 运算，连 as_nanos 都省了！
        self.accum_times[func as usize] = self.start_slot.elapsed();
    }

    // 打印的时候，再把它榨取成带小数点的微秒！
    pub fn print_report(&self) {
        println!("\n=========== ⚡ 单颗 Token 纳秒级高精看板 (μs/Token) ===========");
        
        // 核心解包：拿出纳秒 -> 转 f64 -> 除以圈数 -> 除以 1000 变微秒
        let get_us = |idx: FnIndex| -> f64 {
            (self.accum_times[idx as usize].as_nanos() as f64 ) / 1000.0
        };

        println!("Token Embedding : {:.2} μs", get_us(FnIndex::TokenEmbedding));
        println!("Input RMSNorm   : {:?}", self.accum_times[FnIndex::RmsNormInput as usize]);
        println!("QKV Projection  : {:?}", self.accum_times[FnIndex::QkvProj as usize]);
        println!("RoPE            : {:?}", self.accum_times[FnIndex::rope as usize]);
        println!("Attention Score : {:?}", self.accum_times[FnIndex::AttentionScore as usize]);
        println!("Softmax         : {:?}", self.accum_times[FnIndex::Softmax as usize]);
        println!("Out Projection  : {:?}", self.accum_times[FnIndex::OutProj as usize]);
        println!("MLP Block       : {:?}", self.accum_times[FnIndex::MlpBlock as usize]);
        println!("LM Head         : {:?}", self.accum_times[FnIndex::LmHead as usize]);
        println!("=============================================");
    }
}