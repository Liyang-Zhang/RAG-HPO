<template>
  <section class="results-card">
    <header>
      <h2>分析结果</h2>
      <div class="status-chip">
        <span>共 {{ localItems.length }} 条</span>
      </div>
    </header>
    <table>
      <thead>
        <tr>
          <th>保留</th>
          <th>临床短语</th>
          <th>CHPO 中文</th>
          <th>HPO ID</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(item, index) in localItems" :key="`${item.phrase}-${index}`">
          <td>
            <input
              type="checkbox"
              v-model="item.keep"
              aria-label="是否保留"
            />
          </td>
          <td class="phrase">
            <span>{{ item.phrase }}</span>
          </td>
          <td class="translation">
            <span>{{ item.translation && item.translation.length ? item.translation : "—" }}</span>
          </td>
          <td>
            <input v-model="item.hpo_id" />
          </td>
        </tr>
      </tbody>
    </table>
  </section>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";

import type { PhenotypeRow } from "@/types";

const props = defineProps<{ items: PhenotypeRow[] }>();
const emit = defineEmits<{ (event: "update:items", value: PhenotypeRow[]): void }>();

const localItems = ref<PhenotypeRow[]>([]);

watch(
  () => props.items,
  (value) => {
    localItems.value = value.map((item) => ({ ...item }));
  },
  { immediate: true, deep: true }
);

watch(
  localItems,
  (value) => {
    emit(
      "update:items",
      value.map((item) => ({ ...item }))
    );
  },
  { deep: true }
);
</script>

<style scoped>
.results-card {
  margin-top: 1.5rem;
  background: #fff;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

input[type="checkbox"] {
  width: 18px;
  height: 18px;
}

input[type="text"],
input:not([type]) {
  width: 100%;
  padding: 0.4rem 0.5rem;
  border: 1px solid #cbd5f5;
  border-radius: 6px;
  font: inherit;
}

.phrase {
  max-width: 320px;
}

.translation span {
  display: inline-block;
  max-width: 240px;
  color: #475569;
  word-break: break-word;
}
</style>
