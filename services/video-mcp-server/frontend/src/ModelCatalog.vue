<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import '../model-catalog.js'
import '../model-catalog.css'

const props = defineProps<{ token: string; base: string }>()
const root = ref<HTMLElement>()
let view: { load(): void; destroy(): void; clear(): void } | undefined
onMounted(() => {
  const Catalog = (globalThis as unknown as { VideoCatalogView: new (root: HTMLElement, fetch: () => Promise<unknown>) => NonNullable<typeof view> }).VideoCatalogView
  view = new Catalog(root.value!, async () => {
    const response = await fetch(`${props.base}/api/model-catalog`, { headers: { Authorization: `Bearer ${props.token}` } })
    if (!response.ok) throw new Error('目录读取失败')
    return response.json()
  })
  view.load()
})
watch(() => props.token, () => { view?.clear(); if (props.token) view?.load() })
onBeforeUnmount(() => view?.destroy())
</script>
<template><div ref="root" /></template>
