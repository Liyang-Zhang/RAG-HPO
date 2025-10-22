<template>
  <section class="input-card">
    <header>
      <h2>临床文本</h2>
      <button
        class="secondary"
        type="button"
        :disabled="disabled || !modelValue"
        @click="$emit('clear')"
      >
        清空
      </button>
    </header>
    <textarea
      :value="modelValue"
      :disabled="disabled"
      placeholder="在此粘贴临床描述，提交前请确保已脱敏。"
      @input="onInput"
    />
    <footer>
      <span>字数：{{ modelValue.length }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  modelValue: string;
  disabled?: boolean;
}>();

const emit = defineEmits<{
  (event: "update:modelValue", value: string): void;
  (event: "clear"): void;
}>();

function onInput(event: Event) {
  const target = event.target as HTMLTextAreaElement;
  emit("update:modelValue", target.value);
}
</script>

<style scoped>
.input-card {
  background: #fff;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
  display: grid;
  gap: 0.75rem;
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

footer {
  display: flex;
  justify-content: flex-end;
  color: #64748b;
  font-size: 0.85rem;
}

h2 {
  margin: 0;
  font-size: 1.2rem;
}
</style>
