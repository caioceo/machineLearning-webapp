<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<div class="flex bg-gray-400 h-20">
	<nav class="flex container mx-auto justify-between items-center  w-full">
		<h1>Modelo Machine Learning</h1>
		<h1>Ciências de Dados PUC MG</h1>
		<a href="https://github.com/ICEI-PUC-Minas-PPL-CDIA/ppl-cd-pcd-sist-int-2025-1-desafios-dos-profiss-jr-e-pme-2025-1">Projeto no GitHub</a>
	</nav>
</div>
<div class="container mx-auto">
{@render children()}
</div>
