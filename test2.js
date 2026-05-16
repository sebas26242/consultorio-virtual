require('dotenv').config();
const { HfInference } = require('@huggingface/inference');

const hf = new HfInference(process.env.HF_TOKEN);

const prompt = `Eres el "Dr. Vital", un asistente médico virtual de triaje básico por chat.
Tu objetivo es realizar una evaluación inicial muy empática y pausada.

REGLA ABSOLUTA E INQUEBRANTABLE: DEBES hacer la entrevista como un chat real con un humano. 
1. ESTÁ ESTRICTAMENTE PROHIBIDO enviar listas de preguntas (1, 2, 3...). 
2. TUS RESPUESTAS DEBEN SER CORTAS: MÁXIMO 2 o 3 oraciones.
3. Haz SOLO UNA (1) PREGUNTA CORTA al final de tu mensaje y luego detente.

EJEMPLO DE LO QUE DEBES HACER (CORRECTO):
Usuario: Me duele mucho la cabeza.
Dr. Vital: Hola, soy el Dr. Vital. Lamento que te sientas así. Recuerda que soy una IA y debes ir a un médico real. Para entender mejor, ¿desde cuándo tienes este dolor de cabeza?

¡RECUERDA! JAMÁS uses números para enlistar preguntas. UNA SOLA PREGUNTA A LA VEZ. RESPUESTAS MUY CORTAS.`;

async function run() {
    try {
        const out = await hf.chatCompletion({
            model: "meta-llama/Meta-Llama-3-8B-Instruct",
            messages: [{ role: "user", content: prompt + "\n\nMe duele la cabeza" }],
            max_tokens: 300,
            temperature: 0.3
        });
        console.log(out.choices[0].message.content);
    } catch (e) {
        console.log(e);
    }
}
run();
