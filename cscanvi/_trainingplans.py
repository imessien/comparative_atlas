from scvi.train import SemiSupervisedTrainingPlan


class CLSemiSupervisedTrainingPlan(SemiSupervisedTrainingPlan):
    def log(self, name, value, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        if kwargs.get("on_epoch") and trainer is not None and trainer.world_size > 1:
            kwargs.setdefault("sync_dist", True)
        return super().log(name, value, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module._replay_forward(
            *args,
            **kwargs,
            get_inference_input_kwargs={"full_forward_pass": not self.update_only_decoder},
        )
