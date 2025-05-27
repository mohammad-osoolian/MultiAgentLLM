from utils import parse_json_or_none

class ZeroShotConfig:
    def __init__(self, persona, instruction, output_schema, validator, extractor):
        self.persona = persona
        self.instruction = instruction
        self.output_schema = output_schema
        self.validator = validator
        self.extractor = extractor
        self.dict_output_schema = self.validate_schema()

    def validate_schema(self):
        schema , err = parse_json_or_none(self.output_schema)
        if err:
            raise Exception("output schema is not valid!")
        return schema
        
    def build_system_prompt(self):
        persona = self.persona
        instruction = self.instruction
        output_schema = self.output_schema
        prompt = f'{persona}\n{instruction}\n{output_schema}'
        return prompt

class DebateAgentConfig(ZeroShotConfig):
    def __init__(self, persona, instruction, output_schema, validator, extractor):
        super().__init__(persona, instruction, output_schema, validator, extractor)