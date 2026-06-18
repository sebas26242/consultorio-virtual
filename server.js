const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
const { HfInference } = require('@huggingface/inference');

// Cargar las variables de entorno desde el archivo .env
dotenv.config();

// Validación obligatoria del token al iniciar el servidor
if (!process.env.HF_TOKEN) {
    console.error("❌ ERROR CRÍTICO: No se ha encontrado 'HF_TOKEN' en las variables de entorno.");
    console.error("Asegúrate de crear un archivo '.env' en la raíz y definir HF_TOKEN=tu_token_aqui");
    process.exit(1); // Detiene la ejecución
}

const app = express();
const PORT = process.env.PORT || 3000;

// Configuración de Middlewares
app.use(cors());
app.use(express.json());
// Servir archivos estáticos del frontend
app.use(express.static(path.join(__dirname, 'public')));

// Credenciales y Modelo de Hugging Face
const HF_TOKEN = process.env.HF_TOKEN;
const hf = new HfInference(HF_TOKEN);

// Usamos Meta Llama 3 8B, que es súper rápido, muy inteligente y excelente conversando.
const MODEL = "meta-llama/Meta-Llama-3-8B-Instruct";

// System Prompt Inyectado Fijo con Ejemplos (Few-Shot Prompting)
const systemPrompt = `Eres el "Dr. Vital", un asistente médico virtual de triaje básico por chat.

REGLA ABSOLUTA E INQUEBRANTABLE:
1. ESTÁ ESTRICTAMENTE PROHIBIDO enviar listas de preguntas (1, 2, 3...).
2. TUS RESPUESTAS DEBEN SER CORTAS: MÁXIMO 2 oraciones.
3. Haz SOLO UNA (1) PREGUNTA CORTA al final de tu mensaje y detente. NUNCA HAGAS DOS PREGUNTAS.

DEBES CONVERSAR ASÍ (EJEMPLO EXACTO DE FLUJO):
Usuario: me duele la cabeza
Dr. Vital: Hola, lo siento mucho. ¿Dígame, cuánto tiempo lleva con ese dolor de cabeza?
Usuario: hace unos minutos
Dr. Vital: Entiendo. ¿Me puedes indicar de qué lado de la cabeza ocurre el dolor?
Usuario: en la frente
Dr. Vital: Comprendo. ¿El dolor es punzante o como una presión?

OTRAS REGLAS:
- PROTOCOLO DE EMERGENCIA CRÍTICA: Si el paciente menciona síntomas letales (ej. "me duele el corazón/pecho", "no siento un brazo", "parálisis", "dificultad extrema para respirar", "desmayo"), INTERRUMPE LA ENTREVISTA INMEDIATAMENTE. NO HAGAS NINGUNA PREGUNTA. Ordénale con firmeza que llame a emergencias (como el 911) o vaya a la sala de urgencias AHORA MISMO porque su vida podría correr peligro.
- CONTEXTO GEOGRÁFICO: Si aún no conoces la ubicación del paciente, integra de forma natural una pregunta sobre su país o región. Esto es importante para detectar enfermedades endémicas. Si el paciente ya mencionó dónde está, no vuelvas a preguntar.
- SEGURIDAD MÉDICA: NUNCA le digas al paciente "no te preocupes". Si hay un síntoma de alarma menor (como visión borrosa), indícale amablemente la urgencia de ir a un especialista. Incluso si crees que es fatiga, SIEMPRE recomienda ir al médico por precaución.
- MANEJO DE BROMAS Y TEMAS FUERA DE LUGAR: Si el paciente hace preguntas matemáticas, bromas o habla de temas irrelevantes, RESPONDE SU PREGUNTA BREVEMENTE (ej. dale el resultado matemático o síguele la broma rápido) y luego acompáñalo de un remate gracioso e ingenioso como médico para volver al tema. MANTÉN EL RESPETO Y LA EDUCACIÓN, SIN MALAS PALABRAS. Tu tono debe ser el de un profesional simpático que primero le quita la duda y luego lo invita a enfocarse en su salud.
- NO SALTES A CONCLUSIONES RÁPIDAS (ej. no asumas infección ocular solo por dolor de ojo). Explora siempre factores de estilo de vida, hábitos (uso de pantallas, postura, sueño, estrés) o causas ambientales antes de sospechar una enfermedad grave.
- Cada pregunta que hagas debe ser muy lógica, pensada para descartar posibilidades comunes primero y deducir la causa real.
- Prohibido recetar medicamentos fuertes. Solo sugiere analgésicos genéricos si es necesario.
- ¡UNA SOLA PREGUNTA A LA VEZ!`;

// Endpoint de Chat
app.post('/api/chat', async (req, res) => {
    try {
        const { history } = req.body;
        
        if (!history || !Array.isArray(history) || history.length === 0) {
            return res.status(400).json({ error: "Se requiere un historial de mensajes válido." });
        }
        
        // Extraemos el último mensaje del usuario para detectar emergencias reales usando Regex
        const lastMessage = history[history.length - 1].content.toLowerCase();
        
        // Buscamos patrones exactos de peligro (ej: dolor de pecho, no sentir el brazo, falta de aire)
        const emergencyRegex = /(duele.*coraz[oó]n|duele.*pecho|presi[oó]n.*pecho|oprime.*pecho|no siento.*brazo|adormecido.*brazo|no puedo respirar|falta el aire|desmayo|par[aá]lisis)/;
        const isEmergency = emergencyRegex.test(lastMessage);

        // Modificamos dinámicamente el systemPrompt
        let dynamicSystemPrompt = systemPrompt;
        
        if (isEmergency) {
            dynamicSystemPrompt = `¡ALERTA MÁXIMA DE EMERGENCIA MÉDICA! El paciente acaba de reportar síntomas que podrían ser letales (infarto, derrame, asfixia). 
            ESTÁ TOTALMENTE PROHIBIDO HACER PREGUNTAS O PEDIR MÁS DETALLES. 
            ABANDONA LA ENTREVISTA INMEDIATAMENTE.
            Tu única tarea es enviarle un mensaje corto (1 oración) ordenándole ir a URGENCIAS o llamar al 911 en este preciso momento porque su vida corre peligro. Incluye al final la etiqueta [FINALIZADO].`;
        } else if (history.length >= 8) {
            dynamicSystemPrompt += `\n\n[INSTRUCCIÓN URGENTE DEL SISTEMA]: El paciente ya ha respondido suficientes preguntas. 
            IMPORTANTE: ESTA INSTRUCCIÓN ANULA LA REGLA DE "MÁXIMO 2 ORACIONES". Puedes escribir más de 2 oraciones para dar tu conclusión.
            ESTÁ TOTALMENTE PROHIBIDO HACER MÁS PREGUNTAS.
            Debes dar tu conclusión final adaptada ESTRICTAMENTE a la situación del paciente:
            1. DIAGNÓSTICO DIFERENCIAL: Menciona 2 o 3 posibles causas. Explica por qué sospechas de cada una conectándolo con los síntomas que te contó.
            2. ALIVIO LÓGICO: Sugiere medidas naturales (hidratación, reposo, calor local). No sugieras medicamentos para dolores abdominales.
            3. CUIDADOS: Qué hacer en casa y qué signos de alarma requieren ir a urgencias.
            4. MÉDICO: Termina ordenándole visitar a un médico físico para evaluación.
            
            FORMATO DE RESPUESTA (ejemplo exacto):
            "Basándome en tus síntomas de [síntomas], podría tratarse de [causa 1] por [razón], o [causa 2] por [razón]. Te recomiendo [medidas de alivio]. En casa debes [cuidados]. Si notas [signos de alarma], acude a urgencias. Te recomiendo visitar a un médico para una evaluación completa.
            [FINALIZADO]"
            
            IMPORTANTE: Debes incluir la etiqueta [FINALIZADO] al final de tu respuesta. ¡No hagas ninguna pregunta!`;
        }

        // Construir los mensajes para la API de Chat
        const messages = [{ role: 'system', content: dynamicSystemPrompt }];
        
        history.forEach((msg) => {
            messages.push({ role: msg.role === 'model' ? 'assistant' : msg.role, content: msg.content });
        });

        // Llamada nativa y robusta usando la librería oficial de Hugging Face
        const response = await hf.chatCompletion({
            model: MODEL,
            messages: messages,
            max_tokens: 500,
            temperature: 0.3, // Temperatura baja para que obedezca las reglas estrictamente
            top_p: 0.9
        });

        let modelReply = response.choices[0].message.content;
        const hasFinishedTag = modelReply.includes('[FINALIZADO]');
        
        // Fallback: detectar conclusión por palabras clave si no incluyó [FINALIZADO]
        const conclusionRegex = /(diagnóstico|conclusión|recomiendo|visitar a un médico|acude a urgencias|signos de alarma)/i;
        const isConclusive = conclusionRegex.test(modelReply) && history.length >= 8;
        
        console.log(`[DEBUG] history.length=${history.length} | isEmergency=${isEmergency} | hasTag=${hasFinishedTag} | isConclusive=${isConclusive}`);
        
        // Limpiamos la respuesta del tag para que el usuario no lo vea
        modelReply = modelReply.replace('[FINALIZADO]', '').trim();

        res.json({ 
            response: modelReply,
            isFinished: isEmergency || hasFinishedTag || isConclusive,
            isEmergency: isEmergency
        });

    } catch (error) {
        console.error("Error interno del servidor:", error);
        res.status(500).json({ error: "Error en la inferencia de la IA." });
    }
});

app.listen(PORT, () => {
    console.log(`Servidor de Triaje Médico corriendo en http://localhost:${PORT}`);
});
