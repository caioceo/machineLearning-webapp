<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<div class="flex bg-indigo-500 h-20 text-white">
	<nav class="flex container mx-auto justify-between items-center  w-full">
		<h1>Ciências de Dados - PUC MG</h1>
		<a target="_blank" href="https://github.com/caioceo/machineLearning-webapp">Repositório GitHub</a>
	</nav>
</div>
<div class="container mx-auto">
{@render children()}
</div>
