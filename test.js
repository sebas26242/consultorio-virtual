require('dotenv').config();
const { HfInference } = require('@huggingface/inference');

const hf = new HfInference(process.env.HF_TOKEN);

async function test() {
    try {
        const out = await hf.chatCompletion({
            model: "google/gemma-2-27b-it",
            messages: [{ role: "user", content: "hola" }],
            max_tokens: 50
        });
        console.log(out.choices[0].message);
    } catch (e) {
        console.error("Chat completion error:", e.message);
        
        try {
            console.log("Trying textGeneration...");
            const out2 = await hf.textGeneration({
                model: "google/gemma-2-27b-it",
                inputs: "hola"
            });
            console.log(out2);
        } catch (e2) {
            console.error("Text generation error:", e2.message);
        }
    }
}
test();
