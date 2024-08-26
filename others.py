class OtherAIHandler:
    def process_img(self, img_base64: str, model_name: str):
        raise NotImplementedError("Other AI processing logic is not implemented yet")

    def process_stream_worker(self, stream_url: str, id: str, model_name: str, out_stream_url: str):
        raise NotImplementedError("Other AI streaming logic is not implemented yet")