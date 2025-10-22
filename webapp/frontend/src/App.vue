<template>
  <main>
    <h1>HPO 中文表型标注助手</h1>
    <p class="intro">
      粘贴单条临床描述，系统会自动抽取表型并匹配最可能的 HPO 术语。您可以核对 HPO 编号并导出 CSV。
    </p>

    <PasteInput v-model="note" :disabled="isLoading" @clear="onClear" />

    <section class="actions">
      <button
        class="primary"
        type="button"
        :disabled="isLoading || !note.trim()"
        @click="onAnalyze"
      >
        {{ isLoading ? "分析中..." : "开始分析" }}
      </button>
      <button
        class="secondary"
        type="button"
        :disabled="results.length === 0"
        @click="onExport"
      >
        导出 CSV
      </button>
      <span v-if="runtimeSeconds" class="status-chip">
        用时：{{ runtimeSeconds.toFixed(1) }} 秒 · 保留 {{ keepCount }} 条
      </span>
    </section>

    <section v-if="error" class="error-banner">
      {{ error }}
    </section>

    <ResultTable
      v-if="results.length > 0"
      v-model:items="results"
    />
  </main>
</template>

<script setup lang="ts">
import axios from "axios";
import { computed, ref } from "vue";

import { analyzeClinicalNote } from "@/api/client";
import ResultTable from "@/components/ResultTable.vue";
import PasteInput from "@/components/PasteInput.vue";
import type { AnalyzeResponse, PhenotypeRow } from "@/types";

const note = ref("");
const results = ref<PhenotypeRow[]>([]);
const isLoading = ref(false);
const error = ref("");
const runtimeSeconds = ref(0);

const keepCount = computed(
  () => results.value.filter((item) => item.keep).length
);

async function onAnalyze() {
  if (!note.value.trim()) {
    error.value = "请输入临床描述后再执行分析。";
    return;
  }
  isLoading.value = true;
  error.value = "";
  runtimeSeconds.value = 0;
  results.value = [];
  try {
    const response: AnalyzeResponse = await analyzeClinicalNote(note.value);
    results.value = response.phenotypes;
    runtimeSeconds.value = response.runtime_seconds ?? 0;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      const detail = err.response?.data?.detail;
      error.value = typeof detail === "string" ? detail : "分析失败，请稍后重试。";
    } else {
      error.value = err instanceof Error ? err.message : "分析失败，请稍后重试。";
    }
  } finally {
    isLoading.value = false;
  }
}

function onExport() {
  if (results.value.length === 0) {
    return;
  }
  const header = ["phrase", "translation", "hpo_id", "keep"];
  const rows = results.value.map((item) => [
    item.phrase,
    item.translation ?? "",
    item.hpo_id,
    item.keep ? "yes" : "no",
  ]);
  const csvRows = [header, ...rows];
  const csv = csvRows
    .map((row) =>
      row
        .map((cell) => {
          const value = cell ?? "";
          const escaped = String(value).replace(/"/g, '""');
          return `"${escaped}"`;
        })
        .join(",")
    )
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  link.href = url;
  link.setAttribute("download", `rag-hpo-results-${stamp}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function onClear() {
  note.value = "";
  results.value = [];
  runtimeSeconds.value = 0;
  error.value = "";
}
</script>

<style scoped>
h1 {
  margin: 0 0 0.5rem;
}

.intro {
  margin-top: 0;
  margin-bottom: 1.5rem;
  color: #475569;
}

.actions {
  margin-top: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.actions .status-chip {
  margin-left: auto;
}
</style>
