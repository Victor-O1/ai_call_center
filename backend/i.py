import pyttsx3
import os

def tts_to_wav(
    text: str,
    output_path: str = "output.wav",
    rate: int = 170,
    volume: float = 1.0
):
    engine = pyttsx3.init("sapi5")

    try:
        # Select Microsoft David
        for voice in engine.getProperty("voices"):
            if "david" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        else:
            raise RuntimeError("Microsoft David voice not found")

        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        engine.save_to_file(text, output_path)
        engine.runAndWait()

    finally:
        engine.stop()
        del engine

    return output_path


# Example
tts_to_wav(
       "Talking to an AI call support agent feels like arguing with a polite brick wall that was trained exclusively on corporate apologies. It doesn’t listen—it waits for keywords like a bored bouncer checking IDs. You explain your problem in painful detail, and it responds with the emotional depth of a toaster: “I’m sorry you’re experiencing this issue,” before suggesting the exact same useless step you already said didn’t work—twice. It asks you to reboot things that don’t even have power buttons, loops you through options that lead nowhere, and somehow manages to sound confident while being completely wrong. By the end, you’re not just frustrated—you’re questioning why a system smart enough to parse natural language still can’t grasp a sentence like “I already tried that.” It’s not support; it’s an automated endurance test designed to see how long it takes before you beg for a human."

)
