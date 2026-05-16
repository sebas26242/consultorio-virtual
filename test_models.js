require('dotenv').config();
const { HfInference } = require('@huggingface/inference');

const hf = new HfInference(process.env.HF_TOKEN);

const prompt = `Eres el "Dr. Vital", un asistente médico virtual de triaje básico. REGLA ABSOLUTA: Haz SOLO UNA PREGUNTA CORTA a la vez. PROHIBIDO hacer listas numeradas.`;

async function testModel(modelId) {
    try {
        console.log(`Testing ${modelId}...`);
        const out = await hf.chatCompletion({
            model: modelId,
            messages: [
                { role: "system", content: prompt },
                { role: "user", content: "Me duele la cabeza mucho" }
            ],
            max_tokens: 150
        });
        console.log(`✅ ${modelId}:`, out.choices[0].message.content);
        return true;
    } catch (e) {
        console.log(`❌ Failed ${modelId}:`, e.message);
        return false;
    }
}

async function run() {
    const models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-1.1-7b-it",
        "Qwen/Qwen2.5-72B-Instruct"
    ];
    for (const m of models) {
        await testModel(m);
    }
}
run();
