class EffectManager:
    def __init__(self, cooldown=1000):
        self.effects = []
        self.cooldown = cooldown
        self.last_effect_start_timestamp = 0

    def start_effect(self, effect, time_now):
        if (time_now - self.last_effect_start_timestamp > self.cooldown):

            self.effects.append(effect)
            self.last_effect_start_timestamp = time_now

    def update(self):
        if self.effects:
            for effect in self.effects:
                effect.update()
            self.effects = [effect for effect in self.effects if effect.alive()]

    def draw(self, frame):
        if self.effects:
            for effect in self.effects:
                effect.draw(frame)
